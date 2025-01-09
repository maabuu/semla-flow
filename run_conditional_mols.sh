conda activate evaluation
cd /homes/buttensc/Projects/semla-flow/

files=$(ls predictions/conditional_mol/*/*.sdf)
for file in $files; do

    # nice -n 10 python evaluate_unconditional.py $file

    nice -n 5 python evaluate_conditional_mols.py $file data/conditional_mol/test_first_1000.sdf

done
