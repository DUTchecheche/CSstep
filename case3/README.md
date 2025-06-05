Case 3
===
derived from the basic docking case in the official manual of Autodock Vina: https://autodock-vina.readthedocs.io/en/latest/docking_basic.html

In this case, we aim to de novo design a batch of new lead drug molecules whose SAscore, QED, and binding affinity to the targeted protein all outperform those of the approved anticancer drug imatinib. The target is the kinase domain of the proto-oncogene tyrosine protein kinase c-Abl. The protein is an important target for cancer chemotherapy—in particular, the treatment of chronic myelogenous leukemia. The original crystal structure of the complex of imatinib and kinase c-Abl is available from RCSB PDB (PDB ID:1IEP).

At first, the PyMol software was used to extract the amino acid residues within 6 Å around imatinib as the targeted binding pocket.

Then, without changing any default parameters of CSstep, directly run the following command: 
```
python CSstep.py --output_path .\case3\results --pocket_file .\case3\imatinib_pocket6A.pdb --max_episode 8000
```
The following figure provides a detailed summary of the generation results, and the original output files can also be found in the output_path:

*log_cc.txt*

*initial_mols_cc.txt*

*training_cc.svg*

----

Fig. Overview of the results in this case.

*Further training (more episodes) or using eicosane as an initial molecule should result in better generation results. The currently generated molecules are much smaller in size compared to imatinib (Number of non-hydrogen atoms: 37).*