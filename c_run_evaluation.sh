conda activate evaluation
workdir=/homes/buttensc/Projects/3d_mol_gen_benchmark
cd $workdir

limit=100000000
# limit=50
# limit=1000
nice=20

files=$(ls data/02_postprocessed/*.sdf)
for file in $files; do
    file_name=$(basename $file)
    echo "Evaluating $file_name"

    nice -n $nice python evaluate_molecules.py $file -o $workdir/data/03_evaluation/${file_name%.sdf}.csv --n $limit
done


# without posebusters

files=$(ls data/01_raw/*.sdf)
files_pp=$(ls data/02_postprocessed/*.sdf)
files="$files $files_pp"
for file in $files; do
    file_name=$(basename $file)
    echo "Describing $file_name"

    nice -n $nice python evaluate_molecules.py $file -o $workdir/data/04_description/${file_name%.sdf}_desc.csv --n=$limit --nopb
done
