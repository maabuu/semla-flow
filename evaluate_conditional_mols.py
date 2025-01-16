"""Evaluate molecule-based conditionally generated molecules."""

import argparse
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdmolfiles import MolFromSmarts, MolToSmiles, SDMolSupplier
from rdkit.Chem.rdmolops import RemoveAllHs, RemoveHs, SanitizeMol
from tqdm import tqdm

from tools import (
    compute_ecfp4_tanimoto,
    compute_esp_sim,
    compute_shape_sim,
    compute_smiles,
    compute_sucos,
    get_name,
    get_true_csk_scaffold,
)

logger = logging.getLogger(__name__)


def evaluate_pair(mol_pred: Mol, mol_cond: Mol, name: str) -> dict[str, float]:
    """Evaluate a pair of molecules."""

    results = {}
    try:
        SanitizeMol(mol_pred)
        RemoveAllHs(mol_pred)
        SanitizeMol(mol_cond)
        RemoveAllHs(mol_cond)
        results["tanimoto"] = compute_ecfp4_tanimoto(mol_pred, mol_cond)

        results["sucos"] = compute_sucos(mol_probe=mol_pred, mol_ref=mol_cond)
        results["esp_sim"] = compute_esp_sim(mol_probe=mol_pred, mol_ref=mol_cond)
        results["shape_sim"] = compute_shape_sim(mol_probe=mol_pred, mol_ref=mol_cond)

        results["scaffold_pred"] = get_true_csk_scaffold(mol_pred)
        results["scaffold_cond"] = get_true_csk_scaffold(mol_cond)
        results["scaffold_conserved"] = (
            results["scaffold_pred"] == results["scaffold_cond"]
        )
        results["smiles_pred"] = compute_smiles(mol_pred)
        results["smiles_cond"] = compute_smiles(mol_cond)
        results["num_atoms_pred"] = mol_pred.GetNumHeavyAtoms()
        results["num_atoms_cond"] = mol_cond.GetNumHeavyAtoms()
        results["reference_molecule"] = name
        results["fail"] = False
    except Exception as e:
        results["Error"] = str(e)
        results["fail"] = True
    return results


def parse_arguments():
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description="Evaluate molecules")
    help_line = "Path to SDF file containing predicted molecules."
    parser.add_argument("predicted", type=Path, help=help_line)
    help_line = "Path to SDF file containing conditional molecules."
    parser.add_argument("conditional", type=Path, help=help_line)
    help_line = "Output file."
    parser.add_argument("--output", "-o", type=Path, help=help_line)
    help_line = "Enable debug mode."
    parser.add_argument("--debug", action="store_true", help=help_line)
    return parser.parse_args()


def main(predicted: Path, conditional: Path, output: Path | None = None, debug=False):
    """Evaluate molecules."""

    supplier = SDMolSupplier(str(conditional), removeHs=False, sanitize=False)
    mols_cond = [mol for mol in tqdm(supplier, desc="Loading conditionals")]

    supplier = SDMolSupplier(str(predicted), removeHs=False, sanitize=False)
    total = len(supplier)

    if debug:
        logger.warning("Debug mode enabled.")
        total = 10

    with ProcessPoolExecutor() as executor:
        futures = []
        for i, mol_pred in enumerate(tqdm(supplier, desc="Submitting jobs")):
            if debug and i == total:
                break

            try:
                name = get_name(mol_pred)
                reference_id = int(name.split("_")[-1])
                mol_cond = mols_cond[reference_id]
            except Exception:
                logger.warning("No name found for molecule")
                name = None
                mol_cond = None

            future = executor.submit(evaluate_pair, mol_pred, mol_cond, name)
            futures.append(future)

        results = []
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Collecting jobs"
        ):
            results.append(future.result())

    results_df = pd.DataFrame(results)

    if debug:
        print(results_df)
        return None

    if output is None:
        output = str(predicted).replace(".sdf", "_conditional.csv")
    results_df.to_csv(output, index=False)


if __name__ == "__main__":
    args = parse_arguments()
    main(args.predicted, args.conditional, args.output, debug=args.debug)
