conda activate evaluation
cd /homes/buttensc/Projects/semla-flow/

files=$(ls predictions/conditional_fragments/*/*.sdf)
for file in $files; do

    # nice -n 10 python evaluate_unconditional.py $file

    nice -n 10 python evaluate_conditional_frags.py $file data/conditional_fragments/true_fragments_openbabel_h_sanitize.sdf data/conditional_fragments/reference_mols.sdf  --output ${file%.*}_combined.csv

done

# run the ground truth for linkers
file=data/conditional_fragments/reference_mols_named.sdf
# nice -n 5 python evaluate_unconditional.py $file
nice -n 10 python evaluate_conditional_frags.py $file data/conditional_fragments/true_fragments_openbabel_h_sanitize.sdf data/conditional_fragments/reference_mols.sdf  --output ${file%.*}_combined.csv
