import numpy as np
import pandas as pd


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
