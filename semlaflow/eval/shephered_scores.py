import os
import argparse
from rdkit import Chem
import numpy as np
from rdkit.Chem import SDMolSupplier
from shepherd_score.conformer_generation import optimize_conformer_with_xtb
from rdkit.Chem import SDWriter
from shepherd_score.conformer_generation import embed_conformer

from shepherd_score.evaluations.evaluate import ConsistencyEvalPipeline, ConditionalEvalPipeline
from shepherd_score.container import Molecule



def load_first_mol(sdf_path):
    """Load the first non-None molecule from an SDF file."""
    mol = SDMolSupplier(sdf_path, removeHs=False)[0]
    return mol
    raise ValueError(f"No valid molecule found in {sdf_path}")

def load_all_mols(sdf_path):
    """Load all valid molecules from an SDF file."""
    supplier = SDMolSupplier(sdf_path, removeHs=False)
    return [mol for mol in supplier if mol is not None]


def run_evaluation(reference_sdf, generated_sdf, output_csv):
    ref_mol_rdkit = load_first_mol(reference_sdf)
    gen_mols = load_all_mols(generated_sdf)

    print(f"Loaded {len(gen_mols)} valid generated molecules.")

    ref_molec = Molecule(ref_mol_rdkit, num_surf_points=200, probe_radius=1.2, pharm_multi_vector=False)

    generated_mols = []

    for idx, m in enumerate(gen_mols):
        try:
            generated_mols.append(
            (np.array([a.GetAtomicNum() for a in m.GetAtoms()]), m.GetConformer().GetPositions())
            )
        except Exception as e:
            print(f"Skipping molecule due to MMFF failure: {e}")

    # writer.close()

    cond_pipe = ConditionalEvalPipeline(ref_molec, generated_mols= generated_mols,
                                        condition='all', num_surf_points=200,
                                        pharm_multi_vector=False, solvent=None)
    cond_pipe.evaluate(verbose=True)
    properties_df_cond, global_attr_cond = cond_pipe.to_pandas()

    print(f"Saving results to: {output_csv}")
    global_attr_cond.to_csv(output_csv, index=False)

def main():
    parser = argparse.ArgumentParser(description="Run Shepherd conditional evaluation.")
    parser.add_argument("--reference", required=True, help="Path to reference SDF file (1 molecule)")
    parser.add_argument("--generated", required=True, help="Path to generated SDF file")
    parser.add_argument("--output", required=True, help="Path to output CSV file")

    args = parser.parse_args()

    run_evaluation(args.reference, args.generated, args.output)

if __name__ == "__main__":
    main()