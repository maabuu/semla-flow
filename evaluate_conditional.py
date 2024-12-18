"""Evaluate conditionally generated molecules."""

import argparse
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Generator

import pandas as pd
from posebusters import PoseBusters
from posebusters.modules.sucos import get_sucos_score
from rdkit.Chem import QED, Crippen, Descriptors, Lipinski
from rdkit.Chem.AllChem import GetMorganGenerator
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdMolDescriptors import CalcNumRotatableBonds
from rdkit.Chem.rdmolfiles import MolFromMolBlock, MolToSmiles, SDMolSupplier
from rdkit.Chem.SpacialScore import SPS
from rdkit.DataStructs import TanimotoSimilarity
from rdkit.rdBase import DisableLog
from tqdm import tqdm

fpgen = GetMorganGenerator(radius=2)


def compute_ecfp4_tanimoto(mol_pred: Mol, mol_cond: Mol) -> float:
    """Compute the ECFP4 Tanimoto similarity between two molecules."""
    fingerprint_pred = fpgen.GetSparseCountFingerprint(mol_pred)
    fingerprint_cond = fpgen.GetSparseCountFingerprint(mol_cond)
    return TanimotoSimilarity(fingerprint_pred, fingerprint_cond)


def compute_sucos_score(mol_pred: Mol, mol_cond: Mol) -> float:
    """Compute the SuCOS score between two molecules."""
    return get_sucos_score(mol_pred, mol_cond)


def evaluate_pair(mol_pred: Mol, mol_cond: Mol, name: str) -> dict[str, float]:
    """Evaluate a pair of molecules."""
    results = {}
    try:
        results["tanimoto"] = compute_ecfp4_tanimoto(mol_pred, mol_cond)
        results["sucos"] = compute_sucos_score(mol_pred, mol_cond)
        results["Reference molecule"] = name
    except Exception as e:
        results["Error"] = str(e)
    return results


def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate molecules")
    help_line = "Path to SDF file containing predicted molecules."
    parser.add_argument("predicted", type=Path, help=help_line)
    help_line = "Path to SDF file containing conditional molecules."
    parser.add_argument("conditional", type=Path, help=help_line)
    # default = Path(__file__).parent / "training_smiles.txt"
    # parser.add_argument("--training", type=Path, help=help_line, default=default)
    help_line = "Output file."
    parser.add_argument("--output", "-o", type=Path, help=help_line)
    # help_line = "Total number of molecules."
    # parser.add_argument("--total", type=int, help=help_line, default=0)
    return parser.parse_args()


def evaluate(predicted: Path, conditional: Path, output: Path | None = None):
    """Evaluate molecules."""

    supplier = SDMolSupplier(str(conditional), removeHs=False)
    mols_cond = [mol for mol in tqdm(supplier)]

    results = []
    supplier = SDMolSupplier(str(predicted), removeHs=False)

    futures = []
    with ProcessPoolExecutor() as executor:
        i = 0
        for mol_pred in tqdm(supplier, desc="Submitting jobs"):
            i += 1
            try:
                name = mol_pred.GetProp("_Name")
            except:
                pass
            id_cond = int(name.split("_")[-1])
            mol_cond = mols_cond[id_cond]
            future = executor.submit(evaluate_pair, mol_pred, mol_cond, name)
            futures.append(future)

            # if i % 100 == 0:
            #     break

        desc = "Collecting jobs"
        for future in tqdm(as_completed(futures), total=len(futures), desc=desc):
            results.append(future.result())

    if output is None:
        output = str(predicted).replace(".sdf", "_conditional.csv")
    pd.DataFrame(results).to_csv(output, index=False)


if __name__ == "__main__":
    args = parse_arguments()
    evaluate(args.predicted, args.conditional, args.output)
