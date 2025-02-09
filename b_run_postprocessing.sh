conda activate evaluation
workdir=/homes/buttensc/Projects/3d_mol_gen_benchmark
cd $workdir

files=$(ls $workdir/data/01_raw/*.sdf)
for file in $files; do
    file_name=$(basename $file)
    echo "Postprocessing $file_name"

    nice -n 20 python postprocess_molecules.py $file -o $workdir/data/02_postprocessed/${file_name%.sdf}_pp.sdf
done
