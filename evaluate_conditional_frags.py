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
from rdkit.Chem.rdmolops import AddHs, RemoveHs, SanitizeMol
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


def compute_sucos_score(mol_reference: Mol, mol_probe: Mol) -> float:
    """Compute the SuCOS score between two molecules."""
    return get_sucos_score(mol_reference, mol_probe)


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


def evaluate_pair(
    mol_pred: Mol, mol_frag: Mol, mol_link: Mol, name: str
) -> dict[str, float]:
    """Evaluate a pair of molecules."""

    results = {}
    try:
        SanitizeMol(mol_pred)
        SanitizeMol(mol_frag)
        SanitizeMol(mol_link)
        results["tanimoto_frag"] = compute_ecfp4_tanimoto(mol_pred, mol_frag)
        results["tanimoto_link"] = compute_ecfp4_tanimoto(mol_pred, mol_link)
        results["sucos_frag"] = compute_sucos_score(
            mol_reference=mol_frag, mol_probe=mol_pred
        )
        results["sucos_link"] = compute_sucos_score(
            mol_reference=mol_link, mol_probe=mol_pred
        )
        results["scaffold_true_csk"] = compare_true_csk_scaffolds(mol_pred, mol_link)
        results["scaffold_rdkit_csk"] = compare_rdkit_csk_scaffolds(mol_pred, mol_link)
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


def evaluate(
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

    results = []
    supplier = SDMolSupplier(str(predicted), removeHs=False, sanitize=False)
    total = len(supplier)

    if debug:
        logger.warning("Debug mode enabled.")
        total = 10

    futures = []
    with ProcessPoolExecutor() as executor:
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

        desc = "Collecting jobs"
        for future in tqdm(as_completed(futures), total=len(futures), desc=desc):
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
    evaluate(
        args.predicted, args.fragments, args.linkers, args.output, debug=args.debug
    )
