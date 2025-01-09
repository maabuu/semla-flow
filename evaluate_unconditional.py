"""Evaluate generated molecules individually."""

import argparse
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from posebusters import PoseBusters
from rdkit.Chem import QED, Crippen, Descriptors, Lipinski
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdMolDescriptors import CalcNumRotatableBonds
from rdkit.Chem.rdmolfiles import MolToSmiles, SDMolSupplier
from rdkit.Chem.rdmolops import AddHs, RemoveHs, SanitizeMol
from rdkit.Chem.SpacialScore import SPS
from rdkit.rdBase import DisableLog
from tqdm import tqdm

try:
    from rdkit.Contrib.SA_Score import sascorer
except ImportError:
    sys.path.append(
        os.path.join(os.environ["CONDA_PREFIX"], "share", "RDKit", "Contrib")
    )
    from SA_Score import sascorer


DisableLog("rdApp.*")
logger = logging.getLogger(__name__)
buster = PoseBusters("mol")


def compute_smiles(mol: Mol) -> str:
    """Compute the SMILES string of a molecule."""
    try:
        return MolToSmiles(RemoveHs(mol), canonical=True, allHsExplicit=False)
    except Exception:
        return ""


def get_name(mol: Mol) -> str:
    """Get the name of a molecule."""
    if not hasattr(mol, "HasProp"):
        return ""
    if mol.HasProp("_Name"):
        return mol.GetProp("_Name")
    return ""


def compute_chemical_and_physical_validity(mol: Mol) -> dict[str, bool]:
    """Compute the chemical and physical validity of a molecule."""

    results: dict[str, bool] = buster.bust(mol, full_report=True).iloc[0].to_dict()

    # group checks together
    check_connected = [
        "all_atoms_connected",
    ]
    checks_chemical = [
        "mol_pred_loaded",
        "sanitization",
        "inchi_convertible",
    ]
    checks_physical = [
        "bond_lengths",
        "bond_angles",
        "internal_steric_clash",
        "aromatic_ring_flatness",
        "double_bond_flatness",
        "internal_energy",
    ]
    results |= {
        "connected": all(results[check] for check in check_connected),
        "chemical": all(results[check] for check in checks_chemical),
        "physical": all(results[check] for check in checks_physical),
    }
    chosen = [
        "connected",
        "chemical",
        "physical",
        # "internal_steric_clash",
        # "internal_energy",
        "ensemble_avg_energy",
        "mol_pred_energy",
        "energy_ratio",
    ]

    return {key: results[key] for key in chosen}


def compute_sa_score(mol: Mol) -> float:
    """Compute the synthetic accessibility score of a molecule.
    RDKit blog: https://greglandrum.github.io/rdkit-blog/posts/2023-12-01-using_sascore_and_npscore.html
    Paper reference: https://jcheminf.biomedcentral.com/articles/10.1186/1758-2946-1-8
    """
    try:
        sa_score = sascorer.calculateScore(mol)
        return sa_score
    except Exception:
        return float("nan")


def compute_spacial_score(mol: Mol) -> float:
    """Compute the spacial score of a molecule.
    RDKit reference: https://rdkit.org/docs/source/rdkit.Chem.SpacialScore.html
    Paper reference: https://pubs.acs.org/doi/10.1021/acs.jmedchem.3c00689
    """
    try:
        return float(SPS(mol, normalize=True))
    except Exception:
        return float("nan")


def compute_qed_score(mol: Mol) -> float:
    """Compute the QED score of a molecule.
    RDKit reference: https://www.rdkit.org/docs/source/rdkit.Chem.QED.html
    Paper reference: https://www.nature.com/articles/nchem.1243
    """
    try:
        return float(QED.qed(mol))
    except Exception:
        return float("nan")


def compute_logp(mol: Mol) -> float:
    """Compute the logP of a molecule using the Crippen method.
    RDKit reference: https://www.rdkit.org/docs/source/rdkit.Chem.Crippen.html
    Paper reference: https://pubs.acs.org/doi/10.1021/ci990307l
    """
    try:
        return float(Crippen.MolLogP(mol))
    except Exception:
        return float("nan")


def compute_lipinski_score(mol: Mol) -> float:
    """Compute the Lipinski rule of 5 for a molecule.
    RDKit reference: https://www.rdkit.org/docs/GettingStartedInPython.html#lipinski-rule-of-5
    Paper reference: https://www.sciencedirect.com/science/article/pii/S0169409X00001290
    """
    try:
        rule_1 = Descriptors.ExactMolWt(mol) < 500
        rule_2 = Lipinski.NumHDonors(mol) <= 5
        rule_3 = Lipinski.NumHAcceptors(mol) <= 10
        logp = compute_logp(mol)
        rule_4 = -2 <= logp <= 5
        rule_5 = CalcNumRotatableBonds(mol) <= 10
        return float(sum(rule for rule in [rule_1, rule_2, rule_3, rule_4, rule_5]))
    except Exception:
        return float("nan")


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


def evaluate_one(mol: Mol) -> dict[str, float | int | str]:
    """Evaluate one molecule."""

    results = {}
    results["fail"] = 1
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
        mol = AddHs(mol, addCoords=True)
        results["sa"] = compute_sa_score(mol)
        results["sa_normalized"] = results["sa"] / mol.GetNumHeavyAtoms()
        results["spacial"] = compute_spacial_score(mol)
        results["qed"] = compute_qed_score(mol)
        results["logp"] = compute_logp(mol)
        results["lipinski"] = compute_lipinski_score(mol)
        results["num_heavy"] = mol.GetNumHeavyAtoms()
        results["weight"] = Descriptors.ExactMolWt(mol)
        results["num_rings"] = mol.GetRingInfo().NumRings()
        results["smiles"] = compute_smiles(mol)
        # metrics["num_stero_centers"] = mol.GetNumStereoCenters()
    except Exception as e:
        results["error"] = str(e).replace("\n", " ")
        return results

    return results


def initializer():
    """Initialize the RDKit logger."""
    DisableLog("rdApp.*")


def evaluate(input_file: Path, output_file: Path, debug=False):
    """Evaluate the molecules."""

    DisableLog("rdApp.*")
    supplier = SDMolSupplier(str(input_file), removeHs=False, sanitize=False)
    total = len(supplier)

    if debug:
        logger.warning("Debug mode enabled.")
        total = 200

    with ProcessPoolExecutor(initializer=initializer) as executor:
        futures = []
        for i, mol in enumerate(tqdm(supplier, total=total, desc="Submitting jobs")):
            # if i < 200 or i > 220:
            #     continue
            if debug and i == total:
                break
            futures.append(executor.submit(evaluate_one, mol))
        results = []
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Collecting jobs"
        ):
            results.append(future.result())

    results_df = pd.DataFrame(results)
    if debug:
        print(results_df)
        return None
    output_file = output_file or input_file.with_suffix(".csv")
    results_df.to_csv(output_file, index=False)


if __name__ == "__main__":
    args = parse_arguments()
    evaluate(args.predicted, args.output, debug=args.debug)
