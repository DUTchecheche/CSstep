CSstep: Step-by-step exploration of the chemical space of drug molecules via multi-agent and multi-stage reinforcement learning

CSstep is a drug molecule generation/optimization framework based on the Markov decision process and deep reinforcement learning. CSstep is inspired by Zhou et al. , Jeon et al. , and Fang et al.

By combining a multi-agent joint decision strategy and a multi-stage fine-tuning strategy in this framework, the trade-offs among multiple objective properties during the molecular optimization process and the complete exploration process are visualized in chemical space maps, which is expected to provide more chemical insights for drug discovery and inspire solution ideas for general multi-objective optimization problems. 
CSstep can achieve step-by-step, directed, and intelligent exploration of the drug-like molecular chemical space.

[1] Zhou Z, Kearnes S, Li L, et al. Sci Rep, 2019, 9(1): 10752.

[2] Jeon W, Kim D. Sci Rep, 2020, 10(1): 22104.

[3] Fang Y, Pan X, Shen H-B, et al. Bioinformatics, 2023, 39(4): btad157.

![Fig1](https://github.com/user-attachments/assets/f55f48f6-106d-4ccc-b991-57f4dca0a463)


Usage:

(1) Download this repository and unzip it.

(2) Create and activate a new conda environment:

conda create --name CSstep python=3.7

conda activate CSstep

(3) Install necessary dependencies:

pytorch 1.13.1:

conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia

or

conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 cpuonly -c pytorch

others:

conda install rdkit=2022.9.5 numpy=1.21.6 pandas=1.3.5 matplotlib=3.5.3

options: 

pymol 3.1.0a0, for get_pocket_using_pymol.py

This library can only be used in an environment where the PyMol software is installed.

Download PyMol 3.1: https://www.pymol.org/

(4) Enter the directory where CSstep.py is located:

cd <the directory where CSstep.py is located>

(5) Read the help information:

python CSstep.py -h

(6) Run a De novo molecular generation process:

python CSstep.py --output_path .\test_results --pocket_file .\case1\nilotinib_pocket6A.pdb --max_episode 8000

or (6) Run a molecular optimization process:

python CSstep.py --output_path .\test_results --pocket_file .\case1\nilotinib_pocket6A.pdb --init_smi CC1=C(C=C(C=C1)C(=O)NC2=CC(=CC(=C2)C(F)(F)F)N3C=C(N=C3)C)NC4=NC=CC(=N4)C5=CN=CC=C5 --program_name Nilotinib

(7) Find result files in the output path:

log_cc.txt

initial_mols_cc.txt

training_cc.svg

an example of training_cc.svg:

![training_cc](https://github.com/user-attachments/assets/b959d127-561c-4fe8-a4f0-9c39d4e37108)
