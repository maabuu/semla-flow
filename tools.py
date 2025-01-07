import pandas as pd
import numpy as np


def compute_uniqueness(smiles: list[str]) -> float:
    """Compute the uniqueness of a list of SMILES strings."""
    valid_smiles = [s for s in smiles if s not in {None, "", pd.NA, np.nan}]  # list
    return len(set(valid_smiles)) / len(valid_smiles)


def compute_novelty(smiles: list[str], reference_smiles: set[str]) -> float:
    """How many are not in the test set?"""
    # valid_smiles = set(s for s in smiles if s not in {None, "", pd.NA, np.nan})  # set
    valid_smiles = list(s for s in smiles if s not in {None, "", pd.NA, np.nan})  # list
    return len(
        [smiles for smiles in valid_smiles if smiles not in reference_smiles]
    ) / len(valid_smiles)


def compute_unique_novelty(smiles: list[str], reference_smiles: set[str]) -> float:
    """How many unique new molecules have we generated?"""
    # valid_smiles = set(s for s in smiles if s not in {None, "", pd.NA, np.nan})  # set
    valid_smiles = list(s for s in smiles if s not in {None, "", pd.NA, np.nan})  # list
    return len(set(valid_smiles) - reference_smiles) / len(valid_smiles)
