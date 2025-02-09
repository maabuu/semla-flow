conda activate evaluation
cd /homes/buttensc/Projects/semla-flow/

# files=$(ls data/*/*.sdf)
# files=$(ls data/*/*/*.sdf)
# files=$(ls predictions/unconditional/*/*.sdf)
# for file in $files; do
#     nice -n $nice python evaluate_unconditional.py $file
# done

limit=100000000
# limit=50
limit=1000
nice=20

nice -n $nice python evaluate_unconditional.py predictions/unconditional/eqgat/eqgat_100000_predictions.sdf --n=$limit
nice -n $nice python evaluate_unconditional.py predictions/unconditional/flowmol/flowmol_100000_predictions.sdf --n=$limit
nice -n $nice python evaluate_unconditional.py predictions/unconditional/gcdm/gcdm_100000_predictions.sdf --n=$limit
nice -n $nice python evaluate_unconditional.py predictions/unconditional/geoldm/geoldm_100000_predictions.sdf --n=$limit
nice -n $nice python evaluate_unconditional.py predictions/unconditional/semlaflow/semlaflow_100000_predictions.sdf --n=$limit

nice -n $nice python evaluate_unconditional.py predictions/unconditional/eqgat/eqgat_100000_predictions_optimized.sdf --n=$limit
nice -n $nice python evaluate_unconditional.py predictions/unconditional/flowmol/flowmol_100000_predictions_optimized.sdf --n=$limit
nice -n $nice python evaluate_unconditional.py predictions/unconditional/gcdm/gcdm_100000_predictions_optimized.sdf --n=$limit
nice -n $nice python evaluate_unconditional.py predictions/unconditional/geoldm/geoldm_100000_predictions_optimized.sdf --n=$limit
nice -n $nice python evaluate_unconditional.py predictions/unconditional/semlaflow/semlaflow_100000_predictions_optimized.sdf --n=$limit

nice -n $nice python evaluate_unconditional.py data/unconditional/drugbank/all_structures_3d_optimized.sdf --n=$limit
nice -n $nice python evaluate_unconditional.py data/unconditional/drugbank/all_structures_3d.sdf --n=$limit
nice -n $nice python evaluate_unconditional.py data/unconditional/drugbank/approved_structures_2d.sdf --n=$limit
nice -n $nice python evaluate_unconditional.py data/unconditional/geom-drugs/all.sdf --n=$limit
nice -n $nice python evaluate_unconditional.py data/unconditional/geom-drugs/train.sdf --n=$limit



### just do descriptors

files1=$(ls data/drugbank/*.sdf)
files2=$(ls data/geom-drugs/*/*.sdf)
files3=$(ls predictions/unconditional/*/*.sdf)
files="$files1 $files2 $files3"
for file in $files; do
    nice -n 10 python evaluate_unconditional.py $file -o "${file%.sdf}_descriptors.csv" --nopb
done
