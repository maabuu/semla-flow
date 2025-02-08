conda activate evaluation
cd /homes/buttensc/Projects/semla-flow/

nice -n 20 python improve_molecules.py predictions/unconditional/eqgat/eqgat_100000_predictions.sdf
nice -n 20 python improve_molecules.py predictions/unconditional/gcdm/gcdm_100000_predictions.sdf
nice -n 20 python improve_molecules.py predictions/unconditional/geoldm/geoldm_100000_predictions.sdf
nice -n 20 python improve_molecules.py predictions/unconditional/molflow/molflow_100000_predictions.sdf
nice -n 20 python improve_molecules.py predictions/unconditional/semlaflow/semlaflow_100000_predictions.sdf

nice -n 20 python improve_molecules.py data/unconditional/geom-drugs/train.sdf
nice -n 20 python improve_molecules.py data/unconditional/drugbank/all_structures_3d.sdf
