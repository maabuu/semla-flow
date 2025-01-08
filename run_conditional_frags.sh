#!/bin/bash

cd /homes/buttensc/Projects/semla-flow/
conda activate evaluation

# # Step 1: evaluate the molecules input files individually
# nice -n 20 python evaluate_individually.py data/conditional_fragments/reference_mols.sdf
# nice -n 20 python evaluate_individually.py data/conditional_fragments/true_fragments_openbabel_h_sanitize.sdf

# # Step 2: evaluate the molecules in the prediction files individually
# nice -n 20 python evaluate_individually.py predictions/conditional_fragments/attempt_2/link_replacment_no_h.sdf
# nice -n 20 python evaluate_individually.py predictions/conditional_fragments/attempt_2/link_replacment.sdf

# # Step 3: evaluate the predicted molecules relative to the reference molecules
# nice -n 20 python evaluate_fragment_linking.py predictions/conditional_fragments/attempt_2/link_replacment_no_h.sdf data/fragments/true_fragments_with_h.sdf data/fragments/true_molecules_with_h.sdf  --output predictions/conditional_fragments/attempt_2/link_replacment_no_h_combined.csv
# nice -n 20 python evaluate_fragment_linking.py predictions/conditional_fragments/attempt_2/link_replacment.sdf      data/fragments/true_fragments_with_h.sdf data/fragments/true_molecules_with_h.sdf  --output predictions/conditional_fragments/attempt_2/link_replacment_combined.csv


cd /homes/buttensc/Projects/semla-flow/
conda activate evaluation

files=$(ls predictions/conditional_fragments/attempt_2/*.sdf)
for file in $files; do

    # # assess each molecule individually
    # nice -n 5 python evaluate_individually.py $file &>/dev/null

    # assess each molecule relative to starting fragments
    nice -n 3 python evaluate_fragment_linking.py $file data/conditional_fragments/true_fragments_openbabel_h_sanitize.sdf data/conditional_fragments/reference_mols.sdf  --output ${file%.*}_combined.csv

done


### PREVIOUSLY USED SCRIPTS ###

# Reference molecule and fragments
# files=$(ls data/fragments/*.sdf)
# for file in $files; do
#     # assess each molecule individually
#     nice -n 20 python evaluate_individually.py $file
# done


# files=$(ls predictions/fragment/*.sdf)
# for file in $files; do

#     # # assess each molecule individually
#     # python evaluate_individually.py $file

#     # assess each molecule relative to starting fragments
#     python evaluate_conditional.py $file data/fragments/true_fragments_with_h.sdf --output ${file%.*}_fragment.csv

#     # assess each molecule relative to linker
#     python evaluate_conditional.py $file data/fragments/true_molecules_with_h.sdf --output ${file%.*}_linker.csv
