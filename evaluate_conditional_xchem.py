"""Evaluate fragment screen-based conditionally generated molecules."""

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
    compute_chemical_and_physical_validity,
    compute_esp_sim,
    compute_lipinski_score,
    compute_logp,
    compute_qed_score,
    compute_sa_score,
    compute_shape_sim,
    compute_smiles,
    compute_spacial_score,
    compute_sucos,
    compute_weight,
    get_name,
    get_true_csk_scaffold,
    silence_rdkit,
    split_mol,
)

logger = logging.getLogger(__name__)


def evaluate(mol_pred: Mol, frags: Mol) -> dict[str, float | int | str]:
    """Evaluate a molecule generated to cover a set of fragments."""

    SanitizeMol(mol_pred)
    SanitizeMol(frags)

    fragment_mols = split_mol(frags)
    df = pd.DataFrame({"fragment": fragment_mols})

    # needs Hydrogens

    mol_pred = AddHs(mol_pred, addCoords=True)
    frags = AddHs(frags, addCoords=True)
    df["fragment"] = df["fragment"].apply(lambda f: AddHs(f, addCoords=True))

    df["esp_sim"] = df["fragment"].apply(
        lambda f: compute_esp_sim(mol_probe=mol_pred, mol_ref=f)
    )

    # does not need Hydrogens

    mol_pred = RemoveAllHs(mol_pred)
    frags = RemoveAllHs(frags)
    df["fragment"] = df["fragment"].apply(RemoveAllHs)

    df["shape_sim"] = df["fragment"].apply(
        lambda f: compute_shape_sim(mol_probe=mol_pred, mol_ref=f)
    )
    df["sucos"] = df["fragment"].apply(
        lambda f: compute_sucos(mol_probe=mol_pred, mol_ref=f)
    )

    summary = df[["sucos", "esp_sim", "shape_sim"]].describe().T.reset_index()
    summary = summary.melt(var_name="summary", value_name="value", id_vars="index")
    summary["metric"] = summary["index"] + "_" + summary["summary"]
    results = summary.set_index("metric")["value"].to_dict()

    results["smiles_pred"] = compute_smiles(mol_pred)
    results["scaffold_pred"] = get_true_csk_scaffold(mol_pred)
    return results


def parse_arguments():
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description="Evaluate molecules")
    help_line = "Path to SDF file containing predicted molecules."
    parser.add_argument("predicted", type=Path, help=help_line)
    help_line = "Path to SDF file containing starting fragments."
    parser.add_argument("fragments", type=Path, help=help_line)
    help_line = "Output file."
    parser.add_argument("--output", "-o", type=Path, help=help_line)
    help_line = "Enable debug mode."
    parser.add_argument("--debug", action="store_true", help=help_line)
    return parser.parse_args()


def main(predicted: Path, fragments: Path, output: Path | None = None, debug=False):
    """Evaluate molecules."""

    supplier = SDMolSupplier(str(fragments), removeHs=False, sanitize=False)
    mols_frag = [mol for mol in tqdm(supplier)]

    supplier = SDMolSupplier(str(predicted), removeHs=False, sanitize=False)
    total = len(supplier)

    results = []
    for mol_pred in tqdm(supplier, total=total, desc="Evaluating molecules"):
        # name = get_name(mol_pred)
        # reference_id = int(name.split("_")[-1])
        mol_frag = mols_frag[0]
        result = evaluate(mol_pred, mol_frag)
        results.append(result)

    # with ProcessPoolExecutor() as executor:
    #     futures = []
    #     for i, mol_pred in enumerate(tqdm(supplier, desc="Submitting jobs")):
    #         if debug and i == total:
    #             break

    #         try:
    #             name = get_name(mol_pred)
    #             reference_id = int(name.split("_")[-1])
    #             mol_frag = mols_frag[reference_id]
    #         except Exception:
    #             logger.error("No name found for molecule")
    #             name = None
    #             mol_frag = None

    #         future = executor.submit(evaluate, mol_pred, mol_frag, name)
    #         futures.append(future)

    #     results = []
    #     for future in tqdm(
    #         as_completed(futures), total=len(futures), desc="Collecting jobs"
    #     ):
    #         results.append(future.result())

    results_df = pd.DataFrame(results)
    if debug:
        print(results_df)
        return None
    if output is None:
        output = str(predicted).replace(".sdf", "_conditional.csv")
    results_df.to_csv(output, index=False)


if __name__ == "__main__":
    args = parse_arguments()
    main(args.predicted, args.fragments, args.output, debug=args.debug)
