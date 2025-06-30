"""Calculate the CHEBMLNET activations."""

import argparse
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from fcd import calculate_frechet_distance, canonical_smiles, get_fcd, get_predictions, load_ref_model
from rdkit import RDLogger
from rdkit.Chem.MolStandardize.rdMolStandardize import CleanupParameters, LargestFragmentChooser
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdForceFieldHelpers import MMFFOptimizeMolecule
from rdkit.Chem.rdmolfiles import MolFromMolBlock, MolFromSmiles, MolToMolBlock, MolToSmiles
from rdkit.Chem.rdmolops import AddHs, SanitizeMol
from rdkit.rdBase import LogToPythonLogger
from rich.logging import RichHandler
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TaskProgressColumn, TextColumn, TimeRemainingColumn, track

from tools import hydrate_radicals

logging.basicConfig(level=logging.INFO)
RDLogger.DisableLog("rdApp.*")
np.random.seed(0)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # set gpu

logger = logging.getLogger(__name__)
ProgressBar = partial(
    Progress,
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    MofNCompleteColumn(),
    TaskProgressColumn(),
    TimeRemainingColumn(),
    expand=False,
)

logger.info("Loading model...")
model = load_ref_model()
logger.info("Model loaded.")


def canonical(smi):
    # from fcd.utils import canonical_smiles
    try:
        return MolToSmiles(MolFromSmiles(smi))
    except Exception:
        return None


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description="Evaluate smiles")
    help_line = "Path to evaluation file containing smiles."
    parser.add_argument("smiles", type=Path, help=help_line)

    return parser.parse_args()


def main(input_file: str):
    logger.info("Loading SMILES...")
    smiles = pd.read_csv(input_file, low_memory=False)["smiles"].astype("string").map(canonical).dropna().values
    logger.info(f"Found {len(smiles)} valid SMILES.")
    logger.info("Calculating FCD...")
    activations = get_predictions(model, smiles).astype(np.float64)
    logger.info("Calculating FCD done.")
    logger.info(f"Output shape is {activations.shape}.")
    output_file = Path(input_file).with_suffix(".fcd").as_posix()
    # np.array(activations).tofile(output_file)
    np.save(output_file, activations)
    logger.info(f"FCD saved to {output_file}.")


if __name__ == "__main__":
    args = parse_arguments()
    main(args.smiles)
