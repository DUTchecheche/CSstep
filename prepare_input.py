import re
import numpy as np


def get_atom_feature(atom_i):
    atomic_index_dict = {5: 0, 6: 1, 7: 2, 8: 3, 9: 4, 15: 5, 16: 6, 17: 7, 35: 8, 53: 9}
    atomic_index = np.zeros(len(atomic_index_dict)+1, dtype=int)
    atomic_index[atomic_index_dict.get(atom_i.GetAtomicNum(), len(atomic_index_dict))] = 1
    # 1.Atomic index(type) (one-hot)，11
    atomic_degree_dict = {1: 0, 2: 1, 3: 2, 4: 3}
    atomic_degree = np.zeros(len(atomic_degree_dict)+1, dtype=int)
    atomic_degree[atomic_degree_dict.get(atom_i.GetDegree(), len(atomic_degree_dict))] = 1
    # 2. Atomic degree (one-hot)，5
    total_num_hs_dict = {0: 0, 1: 1, 2: 2, 3: 3}
    total_num_hs = np.zeros(len(total_num_hs_dict)+1, dtype=int)
    total_num_hs[total_num_hs_dict.get(atom_i.GetTotalNumHs(), len(total_num_hs_dict))] = 1
    # 3.Number of hydrogens (one-hot)，5
    formal_charge_dict = {-1: 0, 0: 1, 1: 2}
    formal_charge = np.zeros(len(formal_charge_dict)+1, dtype=int)
    formal_charge[formal_charge_dict.get(atom_i.GetFormalCharge(), len(formal_charge_dict))] = 1
    # 4.Formal charges (one-hot)，4
    if atom_i.IsInRing():
        is_in_ring = np.array([1])
    else:
        is_in_ring = np.array([0])
    # 5. Is in a ring，1
    if atom_i.GetIsAromatic():
        is_aromatic = np.array([1])
    else:
        is_aromatic = np.array([0])
    # 6. Is aromatic，1
    # The six atomic feature descriptors used by Coley et al.
    return np.hstack((atomic_index, atomic_degree, total_num_hs, formal_charge, is_in_ring, is_aromatic))


def pocket2pixel_dict(pocket_file):
    functional_atoms = {'GLY': ['CA'], 'CYS': ['SG'], 'ARG': ['CZ'], 'SER': ['OG'], 'THR': ['OG1'],
                  'LYS': ['NZ'], 'MET': ['SD'], 'ALA': ['CB'], 'LEU': ['CB'], 'ILE': ['CB'],
                  'VAL': ['CB'], 'ASP': ['OD1', 'CG', 'OD2'], 'GLU': ['OE1', 'CD', 'OE2'], 'HIS': ['NE2', 'ND1'],
                  'ASN': ['OD1', 'CG', 'ND2'], 'PRO': ['N', 'CA', 'CB', 'CD', 'CG'], 'GLN': ['OE1', 'CD', 'NE2'],
                  'PHE': ['CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'], 'TRP': ['CD2', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2'],
                  'TRP_2': ['NE1'], 'TYR': ['CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'], 'TYR_2': ['OH']}
    residues_dict = {}
    chain = 'X'
    residue_index = '0'
    residue_name = '000'
    xyz = []
    xyz_2 = []
    Pocket_File = open(pocket_file, 'r')
    for line in Pocket_File:
        if line.startswith('ATOM'):
            now_chain = re.search(r'\w', line[21])[0]
            now_residue_index = re.search(r'\d+', line[22:26])[0]
            now_residue_name = re.search(r'\w+', line[17:20])[0]
            if now_residue_index != residue_index or now_chain != chain:
                if xyz != []:
                    x, y, z = (
                        sum(xyz[i] for i in range(0, len(xyz), 3)) / (len(xyz) // 3),
                        sum(xyz[i] for i in range(1, len(xyz), 3)) / (len(xyz) // 3),
                        sum(xyz[i] for i in range(2, len(xyz), 3)) / (len(xyz) // 3)
                    )
                    residues_dict[chain + residue_index + '_' + residue_name] = [round(x, 3), round(y, 3), round(z, 3)]
                if xyz_2 != []:
                    x, y, z = (
                        sum(xyz_2[i] for i in range(0, len(xyz_2), 3)) / (len(xyz_2) // 3),
                        sum(xyz_2[i] for i in range(1, len(xyz_2), 3)) / (len(xyz_2) // 3),
                        sum(xyz_2[i] for i in range(2, len(xyz_2), 3)) / (len(xyz_2) // 3)
                    )
                    residues_dict[chain + residue_index + '_' + residue_name + '_2'] = [round(x, 3), round(y, 3), round(z, 3)]
                chain = now_chain
                residue_index = now_residue_index
                residue_name = now_residue_name
                xyz = []
                xyz_2 = []
            if re.search(r'\w+', line[12:16])[0] in functional_atoms[residue_name]:
                xyz.append(float(re.search(r'-?\d+\.?\d*e?-?\d*?', line[30:38])[0]))
                xyz.append(float(re.search(r'-?\d+\.?\d*e?-?\d*?', line[38:46])[0]))
                xyz.append(float(re.search(r'-?\d+\.?\d*e?-?\d*?', line[46:54])[0]))
            if residue_name == 'TRP' or residue_name == 'TYR':
                if re.search(r'\w+', line[12:16])[0] in functional_atoms[residue_name+'_2']:
                    xyz_2.append(float(re.search(r'-?\d+\.?\d*e?-?\d*?', line[30:38])[0]))
                    xyz_2.append(float(re.search(r'-?\d+\.?\d*e?-?\d*?', line[38:46])[0]))
                    xyz_2.append(float(re.search(r'-?\d+\.?\d*e?-?\d*?', line[46:54])[0]))
        if line.startswith('END'):
            if xyz != []:
                x, y, z = (
                    sum(xyz[i] for i in range(0, len(xyz), 3)) / (len(xyz) // 3),
                    sum(xyz[i] for i in range(1, len(xyz), 3)) / (len(xyz) // 3),
                    sum(xyz[i] for i in range(2, len(xyz), 3)) / (len(xyz) // 3)
                )
                residues_dict[chain + residue_index + '_' + residue_name] = [round(x, 3), round(y, 3), round(z, 3)]
            if xyz_2 != []:
                x, y, z = (
                    sum(xyz_2[i] for i in range(0, len(xyz_2), 3)) / (len(xyz_2) // 3),
                    sum(xyz_2[i] for i in range(1, len(xyz_2), 3)) / (len(xyz_2) // 3),
                    sum(xyz_2[i] for i in range(2, len(xyz_2), 3)) / (len(xyz_2) // 3)
                )
                residues_dict[chain + residue_index + '_' + residue_name + '_2'] = [round(x, 3), round(y, 3), round(z, 3)]
            break
    Pocket_File.close()
    return residues_dict
    # {'chain+residue_index+'_'+residue_name':[x, y, z],...} eg: {'A40_ASP':[0.1,0.2,0.3],...}


def get_residue_distance(a, b):
    dis = (a[0] - b[0])**2 + (a[1] - b[1])**2 + (a[2] - b[2])**2
    dis = dis**0.5
    return dis


def get_pocket_feature(pixel_dict):
    residues_query = {'ALA': 0, 'ARG': 1, 'ASN': 2, 'ASP': 3, 'CYS': 4,
                           'GLN': 5, 'GLU': 6, 'GLY': 7, 'HIS': 8, 'ILE': 9,
                           'LEU': 10, 'LYS': 11, 'MET': 12, 'PHE': 13, 'PRO': 14,
                           'SER': 15, 'THR': 16, 'TRP': 17, 'TYR': 18, 'VAL': 19,
                           'TYR_2': 20, 'TRP_2': 21}
    # Feature 3: Amino acid type, 22
    Num_HBA_HBD = {'ALA': [0, 0], 'ARG': [1, 2], 'ASN': [1, 1], 'ASP': [1, 1], 'CYS': [1, 1],
                   'GLN': [1, 1], 'GLU': [1, 1], 'GLY': [0, 0], 'HIS': [1, 1], 'ILE': [0, 0],
                   'LEU': [0, 0], 'LYS': [1, 1], 'MET': [1, 0], 'PHE': [0, 0], 'PRO': [0, 0],
                   'SER': [1, 1], 'THR': [1, 1], 'TRP': [0, 0], 'TYR': [0, 0], 'VAL': [0, 0],
                   'TYR_2': [1, 1], 'TRP_2': [0, 1]}
    # Feature 4,5: Number of H-Bond Acceptor and H-Bond donor of  R group of the residue, 3, 3
    Num_rotatable_bonds = {'ALA': 0, 'ARG': 4, 'ASN': 2, 'ASP': 2, 'CYS': 1,
                           'GLN': 3, 'GLU': 3, 'GLY': 0, 'HIS': 2, 'ILE': 2,
                           'LEU': 2, 'LYS': 4, 'MET': 3, 'PHE': 2, 'PRO': 0,
                           'SER': 1, 'THR': 1, 'TRP': 2, 'TYR': 2, 'VAL': 1,
                           'TYR_2': 2, 'TRP_2': 2}
    # Feature 6: Number of rotatable bonds of R group of the residue, 5
    Num_atoms = {'ALA': 1, 'ARG': 1, 'ASN': 3, 'ASP': 3, 'CYS': 1,
                 'GLN': 3, 'GLU': 3, 'GLY': 0, 'HIS': 2, 'ILE': 1,
                 'LEU': 1, 'LYS': 1, 'MET': 1, 'PHE': 6, 'PRO': 5,
                 'SER': 1, 'THR': 1, 'TRP': 6, 'TYR': 6, 'VAL': 1,
                 'TYR_2': 1, 'TRP_2': 1}
    # Feature 7: Number of functional atoms, 7
    hydrophobic = {'ALA': 12, 'ARG': 0, 'ASN': 2, 'ASP': 2, 'CYS': 14,
                 'GLN': 2, 'GLU': 2, 'GLY': 10, 'HIS': 3, 'ILE': 18,
                 'LEU': 16, 'LYS': 1, 'MET': 13, 'PHE': 15, 'PRO': 11,
                 'SER': 8, 'THR': 9, 'TRP': 7, 'TYR': 5, 'VAL': 17,
                 'TYR_2': 4, 'TRP_2': 6}
    # Feature 8: hydrophobicity index rank, ascending order, 19
    pI = {'ALA': 14, 'ARG': 19, 'ASN': 3, 'ASP': 0, 'CYS': 2,
          'GLN': 5, 'GLU': 1, 'GLY': 12, 'HIS': 17, 'ILE': 14,
          'LEU': 13, 'LYS': 18, 'MET': 16, 'PHE': 4, 'PRO': 15,
          'SER': 8, 'THR': 9, 'TRP': 10, 'TYR': 6, 'VAL': 12,
          'TYR_2': 7, 'TRP_2': 11}
    # Feature 9: Isoelectric point rank, ascending order, 20
    alpha = {'ALA': 0, 'ARG': 3, 'ASN': 4, 'ASP': 2, 'CYS': 4,
             'GLN': 1, 'GLU': 0, 'GLY': 5, 'HIS': 2, 'ILE': 1,
             'LEU': 0, 'LYS': 1, 'MET': 0, 'PHE': 1, 'PRO': 5,
             'SER': 3, 'THR': 3, 'TRP': 1, 'TYR': 4, 'VAL': 1,
             'TYR_2': 4, 'TRP_2': 1}
    # Feature 10: Chou-Fasman conformational tendency factor rank for a-helices (strong to weak), 6
    beta = {'ALA': 3, 'ARG': 3, 'ASN': 3, 'ASP': 4, 'CYS': 1,
            'GLN': 1, 'GLU': 5, 'GLY': 4, 'HIS': 3, 'ILE': 0,
            'LEU': 1, 'LYS': 4, 'MET': 1, 'PHE': 1, 'PRO': 5,
            'SER': 4, 'THR': 1, 'TRP': 1, 'TYR': 0, 'VAL': 0,
            'TYR_2': 0, 'TRP_2': 1}
    # Feature 11: Chou-Fasman conformational tendency factor rank for b-sheets (strong to weak), 6
    feature_3 = np.zeros((len(pixel_dict), 22), dtype=int)
    feature_4 = np.zeros((len(pixel_dict), 3), dtype=int)
    feature_5 = np.zeros((len(pixel_dict), 3), dtype=int)
    feature_6 = np.zeros((len(pixel_dict), 5), dtype=int)
    feature_7 = np.zeros((len(pixel_dict), 7), dtype=int)
    feature_8 = np.zeros((len(pixel_dict), 19), dtype=int)
    feature_9 = np.zeros((len(pixel_dict), 20), dtype=int)
    feature_10 = np.zeros((len(pixel_dict), 6), dtype=int)
    feature_11 = np.zeros((len(pixel_dict), 6), dtype=int)
    keys_list = list(pixel_dict.keys())
    for i in range(len(keys_list)):
        residue_name = keys_list[i][(keys_list[i].index('_')+1):]
        feature_3[i][residues_query[residue_name]] = 1
        feature_4[i][Num_HBA_HBD[residue_name][0]] = 1
        feature_5[i][Num_HBA_HBD[residue_name][1]] = 1
        feature_6[i][Num_rotatable_bonds[residue_name]] = 1
        feature_7[i][Num_atoms[residue_name]] = 1
        feature_8[i][hydrophobic[residue_name]] = 1
        feature_9[i][pI[residue_name]] = 1
        feature_10[i][alpha[residue_name]] = 1
        feature_11[i][beta[residue_name]] = 1

    return np.hstack((feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11))
    # 2D array, pocket_size * length of 9 features


def get_pocket_descriptors(max_size, pocket_file):
    pixel_dict = pocket2pixel_dict(pocket_file)

    padding_adj_matrix = np.zeros((max_size, max_size), dtype=int)
    keys_list = list(pixel_dict.keys())
    for i in range(len(keys_list)):
        for j in range(len(keys_list)):
            if i == j:
                padding_adj_matrix[i][j] = 0
            else:
                residue_distance = get_residue_distance(pixel_dict[keys_list[i]], pixel_dict[keys_list[j]])
                if residue_distance < 5.0:
                    padding_adj_matrix[i][j] = 2
                elif 7.0 > residue_distance > 5.0:
                    padding_adj_matrix[i][j] = 1
                else:
                    padding_adj_matrix[i][j] = 0

    feature_1 = np.zeros((len(pixel_dict), 11), dtype=int)
    Num_residues_7A = np.sum(padding_adj_matrix != 0, axis=1)
    Num_residues_7A[Num_residues_7A > 9] = 10
    for i in Num_residues_7A:
        feature_1[i][Num_residues_7A[i]] = 1
    # Feature 1: Num of residues within 7A, 11
    feature_2 = np.zeros((len(pixel_dict), 11), dtype=int)
    Num_residues_5A = np.sum(padding_adj_matrix == 2, axis=1)
    Num_residues_5A[Num_residues_5A > 9] = 10
    for i in Num_residues_5A:
        feature_2[i][Num_residues_5A[i]] = 1
    # Feature 2: Num of residues within 5A, 11
    other_features = get_pocket_feature(pixel_dict)
    # Feature 3-11, 91
    features = np.hstack((feature_1, feature_2, other_features))
    padding_fea_matrix = np.zeros((max_size, 113), dtype=int)
    padding_fea_matrix[:features.shape[0], :features.shape[1]] = features

    return padding_adj_matrix, padding_fea_matrix
