"""Evaluate a set of generated molecules."""

import argparse
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Generator

import pandas as pd
from posebusters import PoseBusters
from rdkit.Chem import QED, Crippen, Descriptors, Lipinski
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdMolDescriptors import CalcNumRotatableBonds
from rdkit.Chem.rdmolfiles import MolFromMolBlock, MolToSmiles
from rdkit.Chem.rdmolops import SanitizeMol
from rdkit.Chem.SpacialScore import SPS
from rdkit.Contrib.SA_Score import sascorer
from rdkit.rdBase import DisableLog
from tqdm import tqdm

logger = logging.getLogger(__name__)
buster = PoseBusters("mol")


def compute_uniquenss(smiles: list[str]) -> float:
    """Compute the uniqueness of a list of SMILES strings."""
    return len(set(smiles)) / len(smiles)


def compute_novelty(smiles: list[str], training_smiles: set[str]) -> float:
    """Compute the novelty of a list of SMILES strings."""
    return len(set(smiles) - training_smiles) / len(smiles)


def compute_chemical_and_physical_validity(mol: Mol) -> dict[str, bool]:
    """Compute the chemical and physical validity of a molecule."""

    results: dict[str, bool] = buster.bust(mol).iloc[0].to_dict()

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
        SanitizeMol(mol)
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
    help_line = "File containing SMILES strings for molecules in training set."
    default = Path(__file__).parent / "training_smiles.txt"
    parser.add_argument("--training", type=Path, help=help_line, default=default)
    help_line = "Output file."
    parser.add_argument("--output", "-o", type=Path, help=help_line)
    help_line = "Total number of molecules."
    parser.add_argument("--total", type=int, help=help_line, default=0)
    return parser.parse_args()


def evaluate_one(mol: Mol) -> dict[str, float]:
    """Evaluate one molecule."""
    metrics = compute_chemical_and_physical_validity(mol)
    metrics["sa"] = compute_sa_score(mol)
    metrics["sa_normalized"] = metrics["sa"] / mol.GetNumHeavyAtoms()
    metrics["spacial"] = compute_spacial_score(mol)
    metrics["qed"] = compute_qed_score(mol)
    metrics["logp"] = compute_logp(mol)
    metrics["lipinski"] = compute_lipinski_score(mol)
    return metrics


def evaluate_batch(mol_blocks: str) -> list[dict]:
    """Evaluate a batch of molecules."""

    DisableLog("rdApp.*")

    results = []
    for mol_block in mol_blocks.split("$$$$\n"):
        mol = MolFromMolBlock(mol_block, sanitize=True)
        if mol is None:
            results.append({"fail": 1})
            continue
        results.append(evaluate_one(mol))
        results[-1]["smiles"] = MolToSmiles(mol)

    return results


def read_file_in_chunks(file_path, chunk_size=10000) -> Generator[str, None, None]:
    """Read molecule file in chunks of molecules."""

    chunk = ""
    molecule_count = 0

    with open(file_path, "r") as file:
        for line in file:
            chunk += line
            if line.startswith("$$$$"):
                molecule_count += 1
                if molecule_count >= chunk_size:
                    yield chunk
                    chunk = ""
                    molecule_count = 0
        # last chunk if not empty
        if chunk:
            yield chunk


def evaluate(
    input_file: Path, training_smiles_file: Path, output_file: Path, total: int = 0
):
    """Evaluate the molecules."""

    results = []
    with ProcessPoolExecutor() as pool:
        futures = [
            pool.submit(evaluate_batch, chunk)
            for chunk in read_file_in_chunks(input_file, 100)
        ]
        for future in tqdm(as_completed(futures), total=len(futures)):
            results.extend(future.result())

    results_df = pd.DataFrame.from_dict(results, orient="columns")

    smiles = results_df.pop("smiles").tolist()
    training_smiles = set(training_smiles_file.read_text().split("\n"))

    metric_uniqueness = compute_uniquenss(smiles)
    metric_novelty = compute_novelty(smiles, training_smiles)

    print(results_df.astype(float).describe())
    print(f"Uniqueness: {metric_uniqueness:.4f}")
    print(f"Novelty: {metric_novelty:.4f}")


if __name__ == "__main__":
    args = parse_arguments()
    evaluate(args.predicted, args.training, args.output, args.total)