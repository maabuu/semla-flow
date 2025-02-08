#!/bin/bash
#SBATCH --account buttensc

#SBATCH --job-name evaluate
#SBATCH --chdir=/vols/opig/users/buttensc/Storage/Projects/semla-flow

#SBATCH --clusters=swan
#S BATCH --nodelist=naga04.cpu.stats.ox.ac.uk
#SBATCH --partition=high-opig-cpu

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=34
#SBATCH --mem=100GB

#S BATCH --array=1-100%100

#SBATCH --output=/vols/opig/users/buttensc/Storage/Projects/semla-flow/slurm/evaluation_individually_%A_%a.out
#SBATCH --error=/vols/opig/users/buttensc/Storage/Projects/semla-flow/slurm/evaluation_individually_%A_%a.err

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
    mamba create -n $env -f environment.yml --yes
else
    # do nothing
    echo "Environment exists"
    mamba env update -n $env -f environment.yml --prune --yes
fi
mamba activate $env

echo "Start task $SLURM_ARRAY_TASK_ID"

cd /vols/opig/users/buttensc/Storage/Projects/semla-flow/predictions

# id=$SLURM_ARRAY_TASK_ID
# files=($(find . -maxdepth 4 -type f -name "*.sdf" | sort))
# input_file=${files[$id]}

# # check if the file exists
# if [ ! -f $input_file ]; then
#     echo "File not found"
#     exit 1
# fi

# # check if output file exists already
# output_file=${input_file%.*}.csv
# if [ -f $output_file ]; then
#     echo "Output file exists"
#     exit 1
# fi

python evaluate_unconditional.py predictions/unconditional/gcdm/gcdm_100000_predictions.sdf --n=$limit
python evaluate_unconditional.py predictions/unconditional/eqgat/eqgat_100000_predictions.sdf --n=$limit
python evaluate_unconditional.py predictions/unconditional/geoldm/geoldm_100000_predictions.sdf --n=$limit
python evaluate_unconditional.py predictions/unconditional/molflow/molflow_100000_predictions.sdf --n=$limit
python evaluate_unconditional.py predictions/unconditional/semlaflow/semlaflow_100000_predictions.sdf --n=$limit

python evaluate_unconditional.py predictions/unconditional/semlaflow/semlaflow_100000_predictions_optimized.sdf --n=$limit
python evaluate_unconditional.py predictions/unconditional/molflow/molflow_100000_predictions_optimized.sdf --n=$limit
python evaluate_unconditional.py predictions/unconditional/geoldm/geoldm_100000_predictions_optimized.sdf --n=$limit
python evaluate_unconditional.py predictions/unconditional/gcdm/gcdm_100000_predictions_optimized.sdf --n=$limit
python evaluate_unconditional.py predictions/unconditional/eqgat/eqgat_100000_predictions_optimized.sdf --n=$limit

# python ../evaluate_unconditional.py $input_file

# echo "Done task $SLURM_ARRAY_TASK_ID"
