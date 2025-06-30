conda activate evaluation
workdir=/homes/buttensc/Projects/3d_mol_gen_benchmark
cd $workdir

limit=100000000
# limit=50
# limit=1000
nice=20

# files=$(ls data/01_raw/*.sdf)
files=$(ls data/02_postprocessed/*.sdf)
files=$(echo "${files[@]}" | tr ' ' '\n' | tac | tr '\n' ' ')
for file in $files; do
    file_name=$(basename $file)
    echo "Evaluating $file_name"

    nice -n $nice python evaluate_molecules.py $file -o $workdir/data/03_evaluation/${file_name%.sdf}.csv --n $limit
done


nice -n 5 python evaluate_molecules.py data/01_raw/geoldm_100000_predictions.sdf -o data/03_evaluation/geoldm_100000_predictions.csv

# nice -n 5 python evaluate_molecules.py data/01_raw/geoldm_100000_predictions.sdf -o data/03_evaluation/geoldm_100000_predictions_1.csv --continue_file=/vols/opig/users/buttensc/Storage/Projects/3d_mol_gen_benchmark/data/03_evaluation/geoldm_100000_predictions.csv
# nice -n 5 python evaluate_molecules.py data/01_raw/geoldm_100000_predictions.sdf -o data/03_evaluation/geoldm_100000_predictions_2.csv --continue_file=/vols/opig/users/buttensc/Storage/Projects/3d_mol_gen_benchmark/data/03_evaluation/geoldm_100000_predictions_1.csv
# nice -n 5 python evaluate_molecules.py data/01_raw/geoldm_100000_predictions.sdf -o data/03_evaluation/geoldm_100000_predictions.csv
# nice -n 10 python evaluate_molecules.py data/01_raw/semlaflow_100000_predictions.sdf -o data/03_evaluation/semlaflow_100000_predictions.csv
# nice -n 10 python evaluate_molecules.py data/01_raw/geom_drugs.sdf -o data/03_evaluation/geom_drugs.csv
# nice -n 10 python evaluate_molecules.py data/02_postprocessed/gcdm_100000_predictions_pp.sdf -o data/03_evaluation/gcdm_100000_predictions_pp.csv


# without posebusters

files=$(ls data/01_raw/*.sdf)
files_pp=$(ls data/02_postprocessed/*.sdf)
# files=$files_pp
files="$files $files_pp"
files=$(echo "${files[@]}" | tr ' ' '\n' | tac | tr '\n' ' ')
for file in $files; do
    file_name=$(basename $file)
    echo "Describing $file_name"

    nice -n $nice python evaluate_molecules.py $file -o $workdir/data/04_description/${file_name%.sdf}_desc.csv --n=$limit --nopb
done
