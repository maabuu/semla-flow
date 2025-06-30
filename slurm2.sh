#!/bin/bash

#SBATCH --account=buttensc

#SBATCH --ntasks=1
#SBATCH --nodes=1

#SBATCH --job-name=evaluation
#SBATCH --cluster=swan
#SBATCH --partition=high-opig-cpu
#S BATCH --nodelist=naga02.cpu.stats.ox.ac.uk

#SBATCH --cpus-per-task=30
#SBATCH --mem=100GB

#SBATCH --error=/vols/opig/users/buttensc/Storage/Slurm/slurm_%N_%j.err   # Writes error messages to this file. %j is jobnumber
#SBATCH --error=/vols/opig/users/buttensc/Storage/Slurm/slurm_%N_%j.err   # Writes error messages to this file. %j is jobnumber

mkdir -p ~/.ssh
rsync -vah /vols/bitbucket/buttensc/Keys/ ~/.ssh

# setup conda
[ ! -d ~/Downloads ] && mkdir ~/Downloads
[ ! -d ~/Applications ] && mkdir ~/Applications
cd ~/Downloads
rm Mambaforge-Linux-x86_64.sh
rm Miniforge3-Linux-x86_64.sh
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -u -p ~/Applications/miniforge
~/Applications/miniforge/bin/mamba init zsh
~/Applications/miniforge/bin/mamba init bash
source ~/.zshrc

# go to dir
cd /vols/opig/users/buttensc/Storage/Projects/3d_mol_gen_benchmark
mamba env create -f environment.yaml --yes
conda activate evaluation

# run
# python evaluate_molecules.py data/01_raw/geom_drugs.sdf -o data/03_evaluation/geom_drugs.csv
# python evaluate_molecules.py data/01_raw/geoldm_100000_predictions.sdf -o data/03_evaluation/geoldm_100000_predictions.csv
# python evaluate_molecules.py data/02_postprocessed/geoldm_100000_predictions_pp.sdf -o data/03_evaluation/geoldm_100000_predictions_pp.csv
# python evaluate_molecules.py data/02_postprocessed/gcdm_100000_predictions_pp.sdf -o data/03_evaluation/gcdm_100000_predictions_pp.csv
