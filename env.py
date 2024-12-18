# Copyright 2023 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import copy
import itertools
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, QED
from SAscore import sascorer
import torch
import torch.nn.functional as F
from prepare_input import get_atom_feature, pocket2pixel_dict, get_pocket_descriptors


def get_atom_valences(atom_types):
  """Creates a list of valences corresponding to atom_types.

  Note that this is not a count of valence electrons, but a count of the
  maximum number of bonds each element will make. For example, passing
  atom_types ['C', 'H', 'O'] will return [4, 1, 2].

  Args:
    atom_types: List of string atom types, e.g. ['C', 'H', 'O'].

  Returns:
    List of integer atom valences.
  """
  periodic_table = Chem.GetPeriodicTable()
  return [
      max(list(periodic_table.GetValenceList(atom_type)))
      for atom_type in atom_types
  ]


def _atom_addition(state, atom_types, atom_valences, atoms_with_free_valence):
  """Computes valid actions that involve adding atoms to the graph.

  Actions:
    * Add atom (with a bond connecting it to the existing graph)

  Each added atom is connected to the graph by a bond. There is a separate
  action for connecting to (a) each existing atom with (b) each valence-allowed
  bond type. Note that the connecting bond is only of type single, double, or
  triple (no aromatic bonds are added).

  For example, if an existing carbon atom has two empty valence positions and
  the available atom types are {'C', 'O'}, this section will produce new states
  where the existing carbon is connected to (1) another carbon by a double bond,
  (2) another carbon by a single bond, (3) an oxygen by a double bond, and
  (4) an oxygen by a single bond.

  Args:
    state: RDKit Mol.
    atom_types: Set of string atom types.
    atom_valences: Dict mapping string atom types to integer valences.
    atoms_with_free_valence: Dict mapping integer minimum available valence
      values to lists of integer atom indices. For instance, all atom indices in
      atoms_with_free_valence[2] have at least two available valence positions.

  Returns:
    Set of string SMILES; the available actions.
  """
  bond_order = {
      1: Chem.BondType.SINGLE,
      2: Chem.BondType.DOUBLE,
      3: Chem.BondType.TRIPLE,
  }
  atom_addition = set()
  for i in bond_order:
    for atom in atoms_with_free_valence[i]:
      for element in atom_types:
        if atom_valences[element] >= i:
          new_state = Chem.RWMol(state)
          idx = new_state.AddAtom(Chem.Atom(element))
          new_state.AddBond(atom, idx, bond_order[i])
          sanitization_result = Chem.SanitizeMol(new_state, catchErrors=True)
          # When sanitization fails
          if sanitization_result:
            continue
          atom_addition.add(Chem.MolToSmiles(new_state))
  return atom_addition


def _bond_addition(state, atoms_with_free_valence, allowed_ring_sizes,
                   allow_bonds_between_rings):
  """Computes valid actions that involve adding bonds to the graph.

  Actions (where allowed):
    * None->{single,double,triple}
    * single->{double,triple}
    * double->{triple}

  Note that aromatic bonds are not modified.

  Args:
    state: RDKit Mol.
    atoms_with_free_valence: Dict mapping integer minimum available valence
      values to lists of integer atom indices. For instance, all atom indices in
      atoms_with_free_valence[2] have at least two available valence positions.
    allowed_ring_sizes: Set of integer allowed ring sizes; used to remove some
      actions that would create rings with disallowed sizes.
    allow_bonds_between_rings: Boolean whether to allow actions that add bonds
      between atoms that are both in rings.

  Returns:
    Set of string SMILES; the available actions.
  """
  bond_orders = [
      None,
      Chem.BondType.SINGLE,
      Chem.BondType.DOUBLE,
      Chem.BondType.TRIPLE,
  ]
  bond_addition = set()
  for valence, atoms in atoms_with_free_valence.items():
    for atom1, atom2 in itertools.combinations(atoms, 2):
      # Get the bond from a copy of the molecule so that SetBondType() doesn't modify the original state.
      bond = Chem.Mol(state).GetBondBetweenAtoms(atom1, atom2)
      new_state = Chem.RWMol(state)
      # Kekulize the new state to avoid sanitization errors; note that bonds
      # that are aromatic in the original state are not modified (this is
      # enforced by getting the bond from the original state with
      # GetBondBetweenAtoms()).
      Chem.Kekulize(new_state, clearAromaticFlags=True)
      if bond is not None:
        if bond.GetBondType() not in bond_orders:
          continue  # Skip aromatic bonds.
        idx = bond.GetIdx()
        # Compute the new bond order as an offset from the current bond order.
        bond_order = bond_orders.index(bond.GetBondType())
        bond_order += valence
        if bond_order < len(bond_orders):
          idx = bond.GetIdx()
          bond.SetBondType(bond_orders[bond_order])
          new_state.ReplaceBond(idx, bond)
        else:
          continue
      # If, do not allow new bonds between atoms already in rings.
      elif (not allow_bonds_between_rings and
            (state.GetAtomWithIdx(atom1).IsInRing() and
             state.GetAtomWithIdx(atom2).IsInRing())):
        continue
      # If the distance between the current two atoms is not in the
      # allowed ring sizes
      elif (allowed_ring_sizes is not None and
            len(Chem.rdmolops.GetShortestPath(
                state, atom1, atom2)) not in allowed_ring_sizes):
        continue
      else:
        new_state.AddBond(atom1, atom2, bond_orders[valence])
      sanitization_result = Chem.SanitizeMol(new_state, catchErrors=True)
      # When sanitization fails
      if sanitization_result:
        continue
      bond_addition.add(Chem.MolToSmiles(new_state))
  return bond_addition


def _bond_removal(state):
  """Computes valid actions that involve removing bonds from the graph.

  Actions (where allowed):
    * triple->{double,single,None}
    * double->{single,None}
    * single->{None}

  Bonds are only removed (single->None) if the resulting graph has zero or one
  disconnected atom(s); the creation of multi-atom disconnected fragments is not
  allowed. Note that aromatic bonds are not modified.

  Args:
    state: RDKit Mol.

  Returns:
    Set of string SMILES; the available actions.
  """
  bond_orders = [
      None,
      Chem.BondType.SINGLE,
      Chem.BondType.DOUBLE,
      Chem.BondType.TRIPLE,
  ]
  bond_removal = set()
  for valence in [1, 2, 3]:
    for bond in state.GetBonds():
      bond = Chem.Mol(state).GetBondBetweenAtoms(bond.GetBeginAtomIdx(),
                                                 bond.GetEndAtomIdx())
      if bond.GetBondType() not in bond_orders:
        continue
      new_state = Chem.RWMol(state)
      Chem.Kekulize(new_state, clearAromaticFlags=True)
      bond_order = bond_orders.index(bond.GetBondType())
      bond_order -= valence
      if bond_order > 0:  # Downgrade this bond.
        idx = bond.GetIdx()
        bond.SetBondType(bond_orders[bond_order])
        new_state.ReplaceBond(idx, bond)
        sanitization_result = Chem.SanitizeMol(new_state, catchErrors=True)
        if sanitization_result:
          continue
        bond_removal.add(Chem.MolToSmiles(new_state))
      elif bond_order == 0:  # Remove this bond entirely.
        atom1 = bond.GetBeginAtom().GetIdx()
        atom2 = bond.GetEndAtom().GetIdx()
        new_state.RemoveBond(atom1, atom2)
        sanitization_result = Chem.SanitizeMol(new_state, catchErrors=True)
        if sanitization_result:
          continue
        smiles = Chem.MolToSmiles(new_state)
        parts = sorted(smiles.split('.'), key=len)
        # We define the valid bond removing action set as the actions
        # that remove an existing bond, generating only one independent
        # molecule, or a molecule and an atom.
        if len(parts) == 1 or len(parts[0]) == 1:
          bond_removal.add(parts[-1])
  return bond_removal


def get_valid_actions(state, atom_types, allow_removal, allow_no_modification, allowed_ring_sizes, allow_bonds_between_rings):
    """Computes the set of valid actions for a given state.

    Args:
      state: String SMILES; the current state. If None or the empty string, we
        assume an "empty" state with no atoms or bonds.
      atom_types: Set of string atom types, e.g. {'C', 'O'}.
      allow_removal: Boolean whether to allow actions that remove atoms and bonds.
      allow_no_modification: Boolean whether to include a "no-op" action.
      allowed_ring_sizes: Set of integer allowed ring sizes; used to remove some
        actions that would create rings with disallowed sizes.
      allow_bonds_between_rings: Boolean whether to allow actions that add bonds
        between atoms that are both in rings.

    Returns:
      Set of string SMILES containing the valid actions (technically, the set of
      all states that are acceptable from the given state).

    Raises:
      ValueError: If state does not represent a valid molecule.
    """
    if not state:
        # Available actions are adding a node of each type.
        return copy.deepcopy(atom_types)
    mol = Chem.MolFromSmiles(state)
    if mol is None:
        raise ValueError('Received invalid state: %s' % state)
    atom_valences = {
        atom_type: get_atom_valences([atom_type])[0]
        for atom_type in atom_types
    }
    atoms_with_free_valence = {}
    for i in range(1, max(atom_valences.values())):
        # Only atoms that allow us to replace at least one H with a new bond are enumerated here.
        atoms_with_free_valence[i] = [
            atom.GetIdx() for atom in mol.GetAtoms() if atom.GetNumImplicitHs() >= i
        ]
    valid_actions = set()
    valid_actions.update(
        _atom_addition(
            mol,
            atom_types=atom_types,
            atom_valences=atom_valences,
            atoms_with_free_valence=atoms_with_free_valence))
    valid_actions.update(
        _bond_addition(
            mol,
            atoms_with_free_valence=atoms_with_free_valence,
            allowed_ring_sizes=allowed_ring_sizes,
            allow_bonds_between_rings=allow_bonds_between_rings))
    if allow_removal:
        valid_actions.update(_bond_removal(mol))
    if allow_no_modification:
        valid_actions.add(Chem.MolToSmiles(mol))
    return list(valid_actions)


def get_fps(smiles, length):
    if smiles is None:
        return np.zeros((length,))
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return np.zeros((length,))
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(molecule, 3, length)
    arr = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(fingerprint, arr)
    return arr


def get_fps_list(smiles_list, length):
    return [get_fps(i, length) for i in smiles_list]


def cal_norm_SAscore(current_smi):
    mol = Chem.MolFromSmiles(current_smi)
    if mol is None:
        return 0.0
    sascore = sascorer.calculateScore(mol)
    norm_sascore = (10 - sascore) / 9
    return round(norm_sascore, 3)


def cal_QED(current_smi):
    mol = Chem.MolFromSmiles(current_smi)
    if mol is None:
        return 0.0
    return round(QED.qed(mol), 3)


# def docking(current_smi):
#     vina_path = r'.\qvina2'
#     os.chdir(vina_path)
#     docking_score = '0'
#     docking_seed = 'error'
#
#     file = open('ligand.smi', 'w')
#     file.write(current_smi)
#     file.close()
#
#     os.system('obabel.exe ligand.smi -O ligand.pdbqt --gen3D --partialcharge Gasteiger -p')
#     os.system('qvina2.exe --config 6m0j.txt --ligand ligand.pdbqt --receptor 6m0j.pdbqt --out docking.pdbqt > docking.log')
#
#     file = open('docking.log', 'r')
#     for line in file:
#         if line.startswith('   1'):
#             docking_score = re.findall(r'-?\d+\.?\d*e?-?\+?\d*', line)[1]
#         if line.startswith('Using random seed:'):
#             docking_seed = re.findall(r'-?\d+\.?\d*e?-?\+?\d*', line)[0]
#     file.close()
#
#     file = open('docking_seed.log', 'a')
#     file.write(docking_score+' '+docking_seed+' '+current_smi+'\n')
#     file.close()
#
#     os.chdir('..')
#     return round(-float(docking_score), 3)


class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_L1 = torch.nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(5, 5))
        self.conv_L2 = torch.nn.Conv2d(in_channels=4, out_channels=8, kernel_size=(3, 3))
        self.pooling_L = torch.nn.MaxPool2d(kernel_size=2)

        self.conv_P1 = torch.nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(5, 5))
        self.conv_P2 = torch.nn.Conv2d(in_channels=4, out_channels=8, kernel_size=(3, 3))
        self.pooling_P = torch.nn.MaxPool2d(kernel_size=(2, 3))

        self.conv1 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(5, 5))
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(3, 3))
        self.pooling = torch.nn.AvgPool2d(kernel_size=2)

        self.fc1 = torch.nn.Linear(128, 16)
        self.fc3 = torch.nn.Linear(16, 1)

    def forward(self, x):
        l = F.relu(self.pooling_L(self.conv_L1(x[0])))
        l = F.relu(self.pooling_L(self.conv_L2(l)))
        p = F.relu(self.pooling_P(self.conv_P1(x[1])))
        p = F.relu(self.pooling_P(self.conv_P2(p)))
        p = torch.nn.functional.pad(p, (0, 4, 0, 0, 0, 0))
        x = torch.cat((l, p), 1)
        x = F.relu(self.pooling(self.conv1(x)))
        x = F.relu(self.pooling(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc3(x)

        return x
    # the QSAR model developed for affinity prediction


def predict_Affinity(model, pocket, current_smi):
    max_size_of_ligand = 80
    try:
        mol = AllChem.MolFromSmiles(current_smi)
        AllChem.SanitizeMol(mol)
        smi = AllChem.MolToSmiles(mol)
        mol = AllChem.MolFromSmiles(smi)
        atoms = mol.GetAtoms()
        num_atoms = sum(1 for atom in atoms if atom.GetAtomicNum() != 1)
        if num_atoms > max_size_of_ligand or num_atoms < 2:
            return 0.0
        else:
            padding_adj_matrix = np.zeros((max_size_of_ligand, max_size_of_ligand), dtype=int)
            adj_matrix = AllChem.GetAdjacencyMatrix(mol)
            padding_adj_matrix[:adj_matrix.shape[0], :adj_matrix.shape[1]] = adj_matrix
            padding_fea_matrix = np.zeros((max_size_of_ligand, 27), dtype=int)
            for i in range(mol.GetNumHeavyAtoms()):
                atom_i = mol.GetAtomWithIdx(i)
                atom_i_features = get_atom_feature(atom_i)
                padding_fea_matrix[i] = atom_i_features
    except:
        return 0.0
    ligand = np.hstack((padding_adj_matrix, padding_fea_matrix))

    with torch.no_grad():
        outputs = model([torch.FloatTensor(ligand).unsqueeze(0).unsqueeze(0), torch.FloatTensor(pocket).unsqueeze(0).unsqueeze(0)])

    return round(float(outputs), 3)


class main_env(object):
    def __init__(self, reward_list, init_smi, pocket_file_path):
        if init_smi and Chem.MolFromSmiles(init_smi):
            self.init_smi = init_smi
        else:
            self.init_smi = 'CC'
            print('An error in initial SMILES! It has been replaced by CC .')
        self.reward_list = reward_list
        self.current_state = self.init_smi
        self.current_step = 0

        self.model = torch.load(r'.\QSAR_parameters.pt')
        self.model.eval()

        max_size_of_pocket = 80
        num_pixel = len(pocket2pixel_dict(pocket_file_path))
        assert 5 <= num_pixel <= max_size_of_pocket, 'The pocket is too large or too small!'
        pocket_adj_matrix, pocket_fea_matrix = get_pocket_descriptors(max_size=max_size_of_pocket,
                                                                      pocket_file=pocket_file_path)
        self.pocket = np.hstack((pocket_adj_matrix, pocket_fea_matrix))
        # load the QSAR model and the pocket input for affinity prediction

    def reset(self):
        self.current_state = self.init_smi
        self.current_step = 0
        return self.current_state

    def step(self, action):
        self.current_state = action
        self.current_step += 1
        return self.current_state

    def reward(self, max_step):
        mol_properties = []
        if 'SAscore' in self.reward_list:
            mol_properties.append(cal_norm_SAscore(self.current_state)*0.9**(max_step-self.current_step))
            # Give a higher reward to the molecule closer to the end of an MDP by using discount factor of 0.9
        if 'QED' in self.reward_list:
            mol_properties.append(cal_QED(self.current_state)*0.9**(max_step-self.current_step))
        if 'Affinity' in self.reward_list:
            mol_properties.append(0.1*predict_Affinity(self.model, self.pocket, self.current_state)*0.9**(max_step-self.current_step))
            # the affinity reward, -lgIC50, -lgKi, or -lgKd, has been reduced to 1/10
        return mol_properties
