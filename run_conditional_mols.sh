cd /homes/buttensc/Projects/semla-flow/
conda activate evaluation

files=$(ls predictions/conditional_mol/attempt_2/*.sdf)
for file in $files; do

    nice -n 20 python evaluate_unconditional.py $file &>/dev/null

    nice -n 20 python evaluate_conditional_mols.py $file data/conditional_mol/test_first_1000.sdf

done
