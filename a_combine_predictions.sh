
# EQGAT - convert and combine xyz files
conda activate evaluation
cd predictions
for i in {0..19}
do
    echo "Processing batch ${i}..."
    obabel batch_${i}/*.xyz -i xyz -o sdf -O combined_batch_${i}.sdf
    echo "Done with batch ${i}..."
done
obabel combined_batch_*.sdf -O ../eqgat_100000_predictions.sdf

# GEOLDM - convert and combine txt files
conda activate evaluation
cd predictions
for i in {0..9}
do
    echo "Processing batch ${i}..."
    obabel analyzed_molecules_${i}/*.txt -i xyz -o sdf -O combined_batch_${i}.sdf
    echo "Done with batch ${i}..."
done
obabel combined_batch_*.sdf -O ../geoldm_100000_predictions.sdf
rm combined_batch_*.sdf
