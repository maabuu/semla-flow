conda activate evaluation
cd /homes/buttensc/Projects/semla-flow/

files=$(ls predictions/unconditional/*/*.sdf)
# files=$(ls data/*/*.sdf)
# files=$(ls data/*/*/*.sdf)
for file in $files; do

    nice -n 3 python evaluate_unconditional.py $file

done
