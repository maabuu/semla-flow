"""Evaluate unconditionally generated molecules."""

import argparse
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdmolfiles import MolFromMolBlock, SDMolSupplier
from rdkit.Chem.rdmolops import AddHs, RemoveAllHs, RemoveHs, SanitizeMol
from tqdm import tqdm

from tools import (
    compute_chemical_and_physical_validity,
    compute_lipinski_score,
    compute_logp,
    compute_qed_score,
    compute_sa_score,
    compute_smiles,
    compute_spacial_score,
    compute_weight,
    get_name,
    silence_rdkit,
)

logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description="Evaluate molecules")
    help_line = "Path to SDF file containing predicted molecules."
    parser.add_argument("predicted", type=Path, help=help_line)
    help_line = "Output file."
    parser.add_argument("--output", "-o", type=Path, help=help_line)
    help_line = "Enable debug mode."
    parser.add_argument("--debug", action="store_true", help=help_line)
    return parser.parse_args()


def evaluate_one(block: str) -> dict[str, float | int | str]:
    """Evaluate one molecule."""

    results = {}
    results["fail"] = 1

    try:
        mol = MolFromMolBlock(block, sanitize=False, removeHs=False, strictParsing=True)
        assert mol is not None
    except Exception as e:
        results["error"] = str(e).replace("\n", " ")
        return results

    try:
        results = compute_chemical_and_physical_validity(mol)
    except Exception as e:
        results["error"] = str(e).replace("\n", " ")
        return results

    results["fail"] = 0

    try:
        results["name"] = get_name(mol)
    except Exception as e:
        results["error"] = str(e).replace("\n", " ")
        return results

    try:
        SanitizeMol(mol)
        # functions with hydrogens
        # AddHs(mol, addCoords=True)

        # functions without hydrogens
        mol = RemoveAllHs(mol)
        results["sa"] = compute_sa_score(mol)
        results["sa_normalized"] = results["sa"] / mol.GetNumHeavyAtoms()
        results["spacial"] = compute_spacial_score(mol)
        results["qed"] = compute_qed_score(mol)
        results["logp"] = compute_logp(mol)
        results["lipinski"] = compute_lipinski_score(mol)
        results["num_heavy"] = mol.GetNumHeavyAtoms()
        results["weight"] = compute_weight(mol)
        results["num_rings"] = mol.GetRingInfo().NumRings()
        results["smiles"] = compute_smiles(mol)

        # metrics["num_stero_centers"] = mol.GetNumStereoCenters()
    except Exception as e:
        results["error"] = str(e).replace("\n", " ")
        return results

    return results


def main(input_file: Path, output_file: Path, debug=False):
    """Evaluate the molecules."""

    silence_rdkit()
    blocks = open(input_file).read().split("$$$$\n")
    total = len(blocks)

    if debug:
        logger.warning("Debug mode enabled.")
        total = 100

    with ProcessPoolExecutor(initializer=silence_rdkit) as executor:
        futures = []
        for i, block in enumerate(tqdm(blocks, total=total, desc="Submitting jobs")):
            # if i < 200 or i > 220:
            #     continue
            if debug and i == total:
                break
            futures.append(executor.submit(evaluate_one, block))

        results = []
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Collecting jobs"
        ):
            try:
                results.append(future.result())
            except Exception as e:
                results.append({"fail": 1, "error": str(e).replace("\n", " ")})

    results_df = pd.DataFrame(results)
    if debug:
        print(results_df)
        return None
    output_file = output_file or input_file.with_suffix(".csv")
    results_df.to_csv(output_file, index=False)


if __name__ == "__main__":
    args = parse_arguments()
    main(args.predicted, args.output, debug=args.debug)
