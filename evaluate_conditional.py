"""Evaluate conditionally generated molecules."""

import argparse
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from posebusters.modules.sucos import get_sucos_score
from rdkit.Chem import AllChem, RemoveStereochemistry
from rdkit.Chem.AllChem import GetMorganGenerator
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdmolfiles import MolFromSmarts, MolToSmiles, SDMolSupplier
from rdkit.Chem.rdmolops import RemoveHs
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.DataStructs import TanimotoSimilarity
from rdkit.rdBase import DisableLog
from tqdm import tqdm

logger = logging.getLogger(__name__)
ecfp4_generator = GetMorganGenerator(radius=2)

PATT = MolFromSmarts("[$([D1]=[*])]")
REPL = MolFromSmarts("[*]")


def get_scaffold(mol, real_bm=True, use_csk=False, use_bajorath=False):
    """Get the scaffold of a molecule."""
    # code from https://github.com/rdkit/rdkit/discussions/6844
    RemoveStereochemistry(mol)  # important for canonization of CSK!
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    if use_bajorath:
        scaffold = AllChem.DeleteSubstructs(scaffold, PATT)
    if real_bm:
        scaffold = AllChem.ReplaceSubstructs(scaffold, PATT, REPL, replaceAll=True)[0]
    if use_csk:
        scaffold = MurckoScaffold.MakeScaffoldGeneric(scaffold)
        if real_bm:
            scaffold = MurckoScaffold.GetScaffoldForMol(scaffold)
    return scaffold


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


def compute_ecfp4_tanimoto(mol_pred: Mol, mol_cond: Mol) -> float:
    """Compute the ECFP4 Tanimoto similarity between two molecules."""
    fingerprint_pred = ecfp4_generator.GetSparseCountFingerprint(mol_pred)
    fingerprint_cond = ecfp4_generator.GetSparseCountFingerprint(mol_cond)
    return TanimotoSimilarity(fingerprint_pred, fingerprint_cond)


def compute_sucos_score(mol_pred: Mol, mol_cond: Mol) -> float:
    """Compute the SuCOS score between two molecules."""
    return get_sucos_score(mol_pred, mol_cond)


def compare_rdkit_csk_scaffolds(mol_pred: Mol, mol_cond: Mol) -> bool:
    """Check whether molecules share the same scaffold."""

    scaffold_pred = MolToSmiles(get_scaffold(mol_pred, real_bm=False, use_csk=True))
    scaffold_cond = MolToSmiles(get_scaffold(mol_cond, real_bm=False, use_csk=True))
    return scaffold_pred == scaffold_cond


def compare_true_csk_scaffolds(mol_pred: Mol, mol_cond: Mol) -> bool:
    """Check whether molecules share the same scaffold."""

    scaffold_pred = MolToSmiles(get_scaffold(mol_pred, real_bm=True, use_csk=True))
    scaffold_cond = MolToSmiles(get_scaffold(mol_cond, real_bm=True, use_csk=True))
    return scaffold_pred == scaffold_cond


def evaluate_pair(mol_pred: Mol, mol_cond: Mol, name: str) -> dict[str, float]:
    """Evaluate a pair of molecules."""

    results = {}
    try:
        results["tanimoto"] = compute_ecfp4_tanimoto(mol_pred, mol_cond)
        results["sucos"] = compute_sucos_score(mol_pred, mol_cond)
        results["scaffold_true_csk"] = compare_true_csk_scaffolds(mol_pred, mol_cond)
        results["scaffold_rdkit_csk"] = compare_rdkit_csk_scaffolds(mol_pred, mol_cond)
        results["smiles_pred"] = compute_smiles(mol_pred)
        results["smiles_cond"] = compute_smiles(mol_cond)
        results["num_atoms_pred"] = mol_pred.GetNumHeavyAtoms()
        results["num_atoms_cond"] = mol_cond.GetNumHeavyAtoms()
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
            name = get_name(mol_pred)
            if name == "":
                logger.warning("No name found for molecule")
                continue
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
