# simplified_eval.py

import argparse
import logging
from pathlib import Path
from typing import Dict, Union, Optional

import pandas as pd
from rdkit.Chem import Crippen, MolFromMolBlock, SDMolSupplier, SanitizeMol, RemoveAllHs
from rdkit.Chem.rdMolDescriptors import CalcNumRotatableBonds
from rdkit.Chem.rdmolops import AddHs, GetMolFrags
from rdkit.rdBase import LogToPythonLogger
from rich.logging import RichHandler
from rich.progress import track

import warnings
warnings.filterwarnings("ignore")

from tools import (
    compute_ghose_filter,
    compute_lipinski_score,
    compute_logp,
    compute_posebusters_validity,
    compute_qed_score,
    compute_sa_score,
    compute_smiles,
    compute_weight,
    count_radicals,
    count_rings,
    compute_sucos,
    protect_from_segmentation_fault,
)

def setup_logging(level=logging.INFO):
    logging.basicConfig(level=level, format="%(message)s", handlers=[RichHandler(rich_tracebacks=True)])
    logger = logging.getLogger("rdkit")
    logger.setLevel(level)
    logger.addHandler(RichHandler(rich_tracebacks=True))
    logger.removeHandler(logger.handlers[0])
    LogToPythonLogger()

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("predicted", type=Path)
    parser.add_argument("--output", "-o", type=Path)
    parser.add_argument("--n", "-n", type=int, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--reference_sdf", type=Path, default=None)
    parser.add_argument("--nopb", action="store_false")
    return parser.parse_args()

def evaluate_one(block: str, pb: bool = True, reference_mol: Optional = None) -> Dict[str, Union[float, int, str]]:
    results = {"name": block.split("\n", 1)[0]}
    try:
        mol = MolFromMolBlock(block, sanitize=False, removeHs=False, strictParsing=False)
        if mol is None:
            results["error"] = "MolFromMolBlock returned None"
            return results
        SanitizeMol(mol)
        mol = RemoveAllHs(mol)

        results["sa"] = compute_sa_score(mol)
        results["qed"] = compute_qed_score(mol)
        results["logp"] = compute_logp(mol)
        results["lipinski"] = compute_lipinski_score(mol)
        results["ghose"] = compute_ghose_filter(mol)
        results["refractivity"] = Crippen.MolMR(mol)
        results["num_heavy"] = mol.GetNumHeavyAtoms()
        results["weight"] = compute_weight(mol)
        results["torsions"] = CalcNumRotatableBonds(mol)
        results["ring_size_count"] = count_rings(mol)
        results["smiles"] = compute_smiles(mol)
        results["connected"] = len(GetMolFrags(mol)) == 1
        results["no_radicals"] = count_radicals(mol) == 0

        if pb:
            protect_from_segmentation_fault(mol)
            results.update(compute_posebusters_validity(mol))

        if reference_mol:
            results["sucos"] = compute_sucos(mol, reference_mol)

        results["errorfree"] = True
    except Exception as e:
        results["error"] = str(e)
    return results

def main():
    args = parse_arguments()
    setup_logging()
    mol_blocks = open(args.predicted).read().split("$$$$\n")
    if args.n:
        mol_blocks = mol_blocks[:args.n]

    reference_mol = None
    if args.reference_sdf:
        for m in SDMolSupplier(str(args.reference_sdf)):
            if m is not None:
                reference_mol = m
                break

    results = []
    for i, block in track(enumerate(mol_blocks), total=len(mol_blocks), description="Evaluating molecules"):
        block = f"molblock_{i:07d}\n" + block.split("\n", 1)[-1]
        result = evaluate_one(block, pb=args.nopb, reference_mol=reference_mol)
        results.append(result)

    df = pd.DataFrame(results)
    if args.debug:
        print(df)
    else:
        out = args.output or args.predicted.with_suffix(".csv")
        df.to_csv(out, index=False)

if __name__ == "__main__":
    main()
