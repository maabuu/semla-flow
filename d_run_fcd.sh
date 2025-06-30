conda activate evaluation
workdir=/homes/buttensc/Projects/3d_mol_gen_benchmark
cd $workdir

nice=20

# files=$(ls data/01_raw/*.sdf)
files=$(ls data/03_evaluation/*.csv)
files=$(echo "${files[@]}" | tr ' ' '\n' | tac | tr '\n' ' ')
for file in $files; do
    file_name=$(basename $file)
    echo "Evaluating $file_name"

    nice -n $nice python fcp.py $file
done
