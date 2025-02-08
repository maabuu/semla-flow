conda activate evaluation
cd /homes/buttensc/Projects/semla-flow/


files=$(ls data/*/*.sdf)
for file in $files; do

    nice -n 20 python evaluate_unconditional.py $file

done

files=$(ls data/*/*/*.sdf)
for file in $files; do

    nice -n 20 python evaluate_unconditional.py $file

done

files=$(ls predictions/unconditional/*/*.sdf)
for file in $files; do

    nice -n 20 python evaluate_unconditional.py $file

done

files=$(ls predictions/conditional_mol/*/*.sdf)
for file in $files; do

    nice -n 20 python evaluate_conditional_mols.py $file data/conditional_mol/test_first_1000.sdf

done

files=$(ls predictions/conditional_fragments/*/*.sdf)
for file in $files; do

    nice -n 20 python evaluate_conditional_frags.py $file data/conditional_fragments/true_fragments_openbabel_h_sanitize.sdf data/conditional_fragments/reference_mols.sdf  --output ${file%.*}_combined.csv

done
