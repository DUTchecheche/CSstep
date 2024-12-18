import os
from pymol import cmd


# This function must be used in a PyMol environment.
def get_pocket(dataset_path, dis=6):
    dis = str(dis)
    targets = os.listdir(dataset_path)
    for i in range(len(targets)):
        target = targets[i]
        try:
            ligand = dataset_path + os.sep + target + os.sep + target + '_ligand.mol2'
            receptor = dataset_path + os.sep + target + os.sep + target + '_protein.pdb'
            cmd.reinitialize()
            cmd.load(ligand, 'ligand')
            cmd.h_add(selection='ligand')
            cmd.load(receptor, 'receptor')
            cmd.remove('resn HOH')
            cmd.select('residues_%sA' % dis, 'byres ligand around %s' % dis)
            cmd.create('%sA_residues' % dis, 'residues_%sA' % dis)
            cmd.save(dataset_path + os.sep + target + os.sep + target + '_pocket%sA.pdb' % dis, '%sA_residues' % dis)
        except:
            print('Unknown error in %s!' % target)
    print('All the pocket files (residues within %sA around the ligand) have been obtained.' % dis)
