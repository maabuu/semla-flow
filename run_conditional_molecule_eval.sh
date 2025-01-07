files=$(ls evaluation/generated/conditional/*.sdf)
for file in $files; do
    python evaluate_conditional.py $file evaluation/truth/test_first_1000.sdf
done
