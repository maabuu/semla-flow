conda activate evaluation
cd /homes/buttensc/Projects/semla-flow/

files=$(ls predictions/conditional_xchem/*.sdf)
for file in $files; do

    # nice -n 10 python evaluate_unconditional.py $file

    nice -n 3 python evaluate_conditional_xchem.py $file /homes/buttensc/Projects/semla-flow/predictions/conditional_xchem/centered_molecules.sdf

done

# nice -n 10 python evaluate_unconditional.py predictions/conditional_xchem/test.sdf


nice -n 10 python evaluate_conditional_xchem.py predictions/conditional_xchem/test.sdf predictions/conditional_xchem/centered_molecules.sdf
