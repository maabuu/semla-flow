"""Utility functions for the evaluations."""

import os
import sys
from typing import Callable

import numpy as np
import pandas as pd
from espsim import GetEspSim, GetShapeSim
from posebusters import PoseBusters
from posebusters.modules.sucos import get_sucos_score
from rdkit.Chem import QED, Crippen, Descriptors, Lipinski, RemoveStereochemistry
from rdkit.Chem.AllChem import DeleteSubstructs, GetMorganGenerator, ReplaceSubstructs
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdMolDescriptors import CalcNumRotatableBonds
from rdkit.Chem.rdmolfiles import MolFromSmarts, MolToSmiles
from rdkit.Chem.rdmolops import (
    AddHs,
    CombineMols,
    GetMolFrags,
    RemoveAllHs,
    RemoveHs,
    SanitizeMol,
)
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.SpacialScore import SPS
from rdkit.DataStructs import TanimotoSimilarity
from rdkit.rdBase import DisableLog

try:
    from rdkit.Contrib.SA_Score import sascorer
except ImportError:
    sys.path.append(
        os.path.join(os.environ["CONDA_PREFIX"], "share", "RDKit", "Contrib")
    )
    from SA_Score import sascorer


def split_mol(mol: Mol, sanitize=True) -> list[Mol]:
    """Split a molecule object into individual molecules."""

    return GetMolFrags(mol, asMols=True, sanitizeFrags=sanitize)


def silence_rdkit():
    """Initialize the RDKit logger."""
    DisableLog("rdApp.*")


def compute_uniqueness(smiles: list[str], total: int = 0) -> float:
    """Compute the uniqueness of a list of SMILES strings."""
    smiles = filter_smiles(smiles)
    total = total or len(smiles)
    if len(smiles) == 0:
        return float("nan")
    return len(set(smiles)) / total


def compute_novelty(
    smiles: list[str], reference_smiles: set[str], total: int = 0
) -> float:
    """How many are not in the reference set?"""
    smiles = filter_smiles(smiles)
    total = total or len(smiles)
    if len(smiles) == 0:
        return float("nan")
    return sum(s not in reference_smiles for s in smiles) / total


def compute_unique_novelty(
    smiles: list[str], reference_smiles: set[str], total: int = 0
) -> float:
    """How many unique new molecules have we generated?"""
    smiles = filter_smiles(smiles)
    total = total or len(smiles)
    if len(smiles) == 0:
        return float("nan")
    return len(set(smiles) - reference_smiles) / total


def filter_smiles(smiles: list[str]) -> list[str]:
    """Filter out invalid SMILES strings."""
    return [s for s in smiles if s not in {None, "", pd.NA, np.nan}]


ecfp4_generator = GetMorganGenerator(radius=2)

PATT = MolFromSmarts("[$([D1]=[*])]")
REPL = MolFromSmarts("[*]")


def get_scaffold(mol, real_bm=True, use_csk=False, use_bajorath=False):
    """Get the scaffold of a molecule."""
    # code from https://github.com/rdkit/rdkit/discussions/6844
    RemoveStereochemistry(mol)  # important for canonization of CSK!
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    if use_bajorath:
        scaffold = DeleteSubstructs(scaffold, PATT)
    if real_bm:
        scaffold = ReplaceSubstructs(scaffold, PATT, REPL, replaceAll=True)[0]
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


def compute_sucos(mol_ref: Mol, mol_probe: Mol) -> float:
    """Compute the SuCOS score between two molecules."""
    return get_sucos_score(mol_ref, mol_probe)


def compute_shape_sim(mol_probe: Mol, mol_ref: Mol) -> float:
    return GetShapeSim(mol_probe=mol_probe, mol_ref=mol_ref)


def compute_esp_sim(mol_probe: Mol, mol_ref: Mol) -> float:
    return GetEspSim(mol_probe=mol_probe, mol_ref=mol_ref)


def get_true_csk_scaffold(mol: Mol) -> str:
    """Get the true CSK scaffold of a molecule."""
    return MolToSmiles(get_scaffold(mol, real_bm=True, use_csk=True))


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


def compute_weight(mol: Mol) -> Mol:
    """Compute the molecular weight of a molecule."""
    return Descriptors.ExactMolWt(mol)


buster = PoseBusters("mol")


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
