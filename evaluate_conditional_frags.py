"""Evaluate fragment-based conditionally generated molecules."""

import argparse
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdmolfiles import SDMolSupplier
from rdkit.Chem.rdmolops import AddHs, RemoveAllHs, SanitizeMol
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


def evaluate_pair(
    mol_pred: Mol, mol_frag: Mol, mol_link: Mol, name: str
) -> dict[str, float]:
    """Evaluate a pair of molecules."""

    results = {}
    try:
        SanitizeMol(mol_pred)
        SanitizeMol(mol_frag)
        SanitizeMol(mol_link)

        # needs Hydrogens

        mol_pred = AddHs(mol_pred, addCoords=True)
        mol_frag = AddHs(mol_frag, addCoords=True)
        mol_link = AddHs(mol_link, addCoords=True)

        results["esp_sim_frag"] = compute_esp_sim(mol_probe=mol_frag, mol_ref=mol_pred)
        results["esp_sim_link"] = compute_esp_sim(mol_probe=mol_link, mol_ref=mol_pred)

        # does not need Hydrogens

        mol_pred = RemoveAllHs(mol_pred)
        mol_frag = RemoveAllHs(mol_frag)
        mol_link = RemoveAllHs(mol_link)

        results["tanimoto_frag"] = compute_ecfp4_tanimoto(mol_pred, mol_frag)
        results["tanimoto_link"] = compute_ecfp4_tanimoto(mol_pred, mol_link)
        results["shape_sim_frag"] = compute_shape_sim(
            mol_probe=mol_frag, mol_ref=mol_pred
        )
        results["shape_sim_link"] = compute_shape_sim(
            mol_probe=mol_link, mol_ref=mol_pred
        )
        results["sucos_frag"] = compute_sucos(mol_probe=mol_frag, mol_ref=mol_pred)
        results["sucos_link"] = compute_sucos(mol_probe=mol_link, mol_ref=mol_pred)

        results["scaffold_pred"] = get_true_csk_scaffold(mol_pred)
        results["scaffold_link"] = get_true_csk_scaffold(mol_link)
        results["scaffold_conserved"] = (
            results["scaffold_pred"] == results["scaffold_link"]
        )

        results["smiles_pred"] = compute_smiles(mol_pred)
        results["smiles_link"] = compute_smiles(mol_link)
        results["num_atoms_pred"] = mol_pred.GetNumHeavyAtoms()
        results["num_atoms_frag"] = mol_frag.GetNumHeavyAtoms()
        results["num_atoms_link"] = mol_link.GetNumHeavyAtoms()
        results["Reference molecule"] = name
        results["fail"] = 0
    except Exception as e:
        results["fail"] = 1
        results["Error"] = str(e).replace("\n", " ")
    return results


def parse_arguments():
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description="Evaluate molecules")
    help_line = "Path to SDF file containing predicted molecules."
    parser.add_argument("predicted", type=Path, help=help_line)
    help_line = "Path to SDF file containing starting fragments."
    parser.add_argument("fragments", type=Path, help=help_line)
    help_line = "Path to SDF file containing true linkers."
    parser.add_argument("linkers", type=Path, help=help_line)
    help_line = "Output file."
    parser.add_argument("--output", "-o", type=Path, help=help_line)
    help_line = "Enable debug mode."
    parser.add_argument("--debug", action="store_true", help=help_line)
    return parser.parse_args()


def main(
    predicted: Path,
    fragments: Path,
    linkers: Path,
    output: Path | None = None,
    debug=False,
):
    """Evaluate molecules."""

    supplier = SDMolSupplier(str(fragments), removeHs=False, sanitize=False)
    mols_frag = [mol for mol in tqdm(supplier)]

    supplier = SDMolSupplier(str(linkers), removeHs=False, sanitize=False)
    mols_link = [mol for mol in tqdm(supplier)]

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
                mol_frag = mols_frag[reference_id]
                mol_link = mols_link[reference_id]
            except Exception:
                logger.warning("No name found for molecule")
                name = None
                mol_frag = None
                mol_link = None

            future = executor.submit(evaluate_pair, mol_pred, mol_frag, mol_link, name)
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
    main(args.predicted, args.fragments, args.linkers, args.output, debug=args.debug)
