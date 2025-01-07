# # Reference molecule and fragments
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

# done


files=$(ls predictions/fragment/*.sdf)
for file in $files; do

    # # assess each molecule individually
    # python evaluate_individually.py $file

    # assess each molecule relative to starting fragments
    python evaluate_fragment_linking.py $file data/fragments/true_fragments_with_h.sdf data/fragments/true_molecules_with_h.sdf  --output ${file%.*}_combined.csv

done
