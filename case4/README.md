Case 4
===
In this case, we aim to optimize a known GPCR antagonist, ZM241385 (Jaakola, et al., 2008. The 2.6 angstrom crystal structure of a human A2A adenosine receptor bound to an antagonist. Science 322, 1211–1217.). The adenosine class of heterotrimeric guanine nucleotide-binding protein (G protein)-coupled receptors (GPCRs) mediates the important role of extracellular adenosine in many physiological processes. Development of more selective compounds for adenosine receptor subtypes could provide a class of therapeutics for treating numerous human maladies, such as pain, Parkinson’s disease, Huntington’s disease, asthma, seizures, and many other neurological disorders. 4-(2-[7-amino-2-(2-furyl)-[1,2,4]triazolo[2,3-a][1,3,5]triazin-5-ylamino]ethyl)-phenol (ZM241385) has been reported as a subtype-selective high-affinity antagonist. The original crystal structure of the complex of ZM241385 and the human A2A Adenosine Receptor is available from RCSB PDB (PDB ID:3EML). 

At first, the PyMol software was used to extract the amino acid residues within 6 Å around ZM241385 as the targeted binding pocket. The SMILES of ZM241385 was obtained using RDKit. 

Then, using the following command to run a molecular optimization task:
```
python CSstep.py --output_path .\case4\results --pocket_file .\case4\ZM241385_pocket6A.pdb --init_smi Nc1nc(NCCc2ccc(O)cc2)nc2nc(-c3ccco3)nn12 --program_name ZM241385
```
The following figure provides a detailed summary of the generation results, and the original output files can also be found in the output_path:

*log_ZM241385.txt*

*initial_mols_ZM241385.txt*

*training_ZM241385.svg*

----
![results](https://github.com/user-attachments/assets/d02ab036-e585-4868-819c-102a9724260d)
Fig. Overview of the results in this case.
