# individual assessment
conda activate evaluation
cd /homes/buttensc/Projects/semla-flow/

nice -n 20 python evaluate_individually.py predictions/unconditional/eqgat/eqgat_100000_predictions.sdf

nice -n 20 python evaluate_individually.py predictions/unconditional/geoldm/geoldm_100000_predictions.sdf
