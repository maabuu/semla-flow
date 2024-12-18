#!/bin/bash
#SBATCH --account buttensc

#SBATCH --job-name evaluate
#SBATCH --chdir=/vols/opig/users/buttensc/Storage/Projects/semla-flow

#SBATCH --clusters=swan
#SBATCH --nodelist=naga05.cpu.stats.ox.ac.uk
#SBATCH --partition=high-opig-cpu

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=34
#SBATCH --mem=340GB

#SBATCH --array=1-10%1

#SBATCH --output=/vols/opig/users/buttensc/Storage/Projects/semla-flow/slurm/generate_10000_%A_%a.out
#SBATCH --error=/vols/opig/users/buttensc/Storage/Projects/semla-flow/slurm/generate_10000_%A_%a.err

# setup directories
[ ! -d ~/Downloads ] && mkdir ~/Downloads
[ ! -d ~/Applications ] && mkdir ~/Applications
[ ! -d ~/Projects ] && mkdir ~/Projects
[ ! -d ~/Network ] && ln -s /vols/opig/users/buttensc/Storage ~/Network

# setup mamba - https://github.com/conda-forge/miniforge
prefix=/homes/buttensc/Applications/miniforge
source $prefix/etc/profile.d/conda.sh
source $prefix/etc/profile.d/mamba.sh
if ! command -v conda &> /dev/null; then
    cd ~/Downloads
    rm -rf ~/Applications/miniforge
    rm Miniforge3-*
    curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
    bash Miniforge3-$(uname)-$(uname -m).sh -b -u -p $prefix
    source $prefix/etc/profile.d/conda.sh
    source $prefix/etc/profile.d/mamba.sh
    conda activate
else
    source $prefix/etc/profile.d/conda.sh
    source $prefix/etc/profile.d/mamba.sh
    conda activate
fi

# update env
env=evaluation
if [ ! -d "$prefix/envs/$env" ]; then
    mamba create -n $env python=3.12 plotly rdkit pandas jupyterlab tqdm posebusters --yes
else
    # do nothing
    echo "Environment exists"
fi
mamba activate $env


echo "Start task $SLURM_ARRAY_TASK_ID"

cd /vols/opig/users/buttensc/Storage/Projects/semla-flow/predictions

id=$SLURM_ARRAY_TASK_ID
files=($(find . -maxdepth 1 -type f -name "*.sdf" | sort))
input_file=${files[$id]}

# check if the file exists
if [ ! -f $input_file ]; then
    echo "File not found"
    exit 1
fi

python ../evaluate_molecules.py $input_file

echo "Done task $SLURM_ARRAY_TASK_ID"
