"""Evaluate unconditionally generated molecules."""

import argparse
import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from functools import partial
from pathlib import Path

import pandas as pd
from rdkit.Chem import QED, Crippen
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdMolDescriptors import CalcNumRotatableBonds
from rdkit.Chem.rdmolfiles import MolFromMolBlock
from rdkit.Chem.rdmolops import AddHs, CombineMols, GetMolFrags, RemoveAllHs, RemoveHs, SanitizeMol
from rdkit.rdBase import LogToPythonLogger
from rich.logging import RichHandler
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TaskProgressColumn, TextColumn, TimeRemainingColumn, track

from tools import (
    compute_chemical_and_physical_validity,
    compute_ghose_filter,
    compute_lipinski_score,
    compute_logp,
    compute_qed_score,
    compute_sa_score,
    compute_smiles,
    compute_spacial_score,
    compute_weight,
    count_radicals,
    count_rings,
    get_name,
)

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


def setup_logging(level=logging.ERROR) -> int:
    """Set up logging."""

    logging.basicConfig(level=level, format="%(message)s", datefmt="[%X]", handlers=[RichHandler(markup=True, rich_tracebacks=True)])

    logger = logging.getLogger("rdkit")
    logger.setLevel(level)
    logger.addHandler(RichHandler(rich_tracebacks=True))
    logger.removeHandler(logger.handlers[0])
    LogToPythonLogger()

    return level


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description="Evaluate molecules")
    help_line = "Path to SDF file containing predicted molecules."
    parser.add_argument("predicted", type=Path, help=help_line)
    help_line = "Output file."
    parser.add_argument("--output", "-o", type=Path, help=help_line)
    help_line = "Run on first N molecules only."
    parser.add_argument("--n", "-n", type=int, help=help_line, default=None)
    help_line = "Disable PB checks."
    parser.add_argument("--nopb", action="store_false", help=help_line)
    help_line = "Logging level."
    parser.add_argument("--logging", "-l", type=int, help=help_line, default=logging.CRITICAL)
    help_line = "Enable debug mode."
    parser.add_argument("--debug", action="store_true", help=help_line)
    return parser.parse_args()


def evaluate_one(block: str, pb=True) -> dict[str, float | int | str]:
    """Evaluate one molecule."""

    results = {}
    results["name"] = block.split("\n", 2)[0]
    results["loads"] = 0
    results["sanitizes"] = 0
    results["connected"] = 0
    results["all_hydrogens"] = 0
    results["no_radicals"] = 0
    results["errorfree"] = 0

    try:
        mol = MolFromMolBlock(block, sanitize=False, removeHs=False, strictParsing=False)
        assert mol is not None
        results["loads"] = 1
    except Exception as e:
        results["error"] = str(e).replace("\n", " ")
        return results

    try:
        mol = MolFromMolBlock(block, sanitize=False, removeHs=False, strictParsing=False)
        mol.RemoveAllConformers()
        SanitizeMol(mol, catchErrors=False)
        results["sanitizes"] = 1
        results["all_hydrogens"] = int((AddHs(mol).GetNumAtoms() - mol.GetNumAtoms()) == 0)
        results["no_radicals"] = int(count_radicals(mol) == 0)
        results["connected"] = int(len(GetMolFrags(mol, asMols=False, sanitizeFrags=False)) == 1)
    except Exception as e:
        results["error"] = str(e).replace("\n", " ")
        return results

    try:
        # most RDKit functions assume that Hydrogens are implicit
        mol = MolFromMolBlock(block, sanitize=False, removeHs=False, strictParsing=False)
        SanitizeMol(mol, catchErrors=False)
        mol = RemoveAllHs(mol)
        results["sa"] = compute_sa_score(mol)
        results["sa_normalized"] = results["sa"] / mol.GetNumHeavyAtoms()
        results["spacial"] = compute_spacial_score(mol)
        results["qed"] = compute_qed_score(mol)
        results["logp"] = compute_logp(mol)
        results["lipinski"] = compute_lipinski_score(mol)
        results["ghose"] = compute_ghose_filter(mol)
        results["refractivity"] = Crippen.MolMR(mol)
        results["num_heavy"] = mol.GetNumHeavyAtoms()
        results["num_total"] = mol.GetNumAtoms()
        results["weight"] = compute_weight(mol)
        results["num_rings"] = mol.GetRingInfo().NumRings()
        results["num_torsions"] = CalcNumRotatableBonds(mol)  # uses default values
        results["ring_size_count"] = count_rings(mol)
        results["smiles"] = compute_smiles(mol)
        # results["num_stero_centers"] = mol.GetNumStereoCenters()
    except Exception as e:
        results["error"] = str(e).replace("\n", " ")
        return results

    if pb:
        try:
            mol = MolFromMolBlock(block, sanitize=False, removeHs=False, strictParsing=False)
            results |= compute_chemical_and_physical_validity(mol)
        except Exception as e:
            results["error"] = str(e).replace("\n", " ")
            return results

    results["errorfree"] = 1
    return results


def main(input_file: Path, output_file: Path, n: int | None = None, timeout: int | None = None, debug: bool = False, pb=True):
    """Evaluate the molecules."""

    blocks = open(input_file).read().rstrip().rstrip("\n").rstrip("\n").rstrip("$$$$").split("$$$$\n")

    if n and len(blocks) > n:
        logger.warning("Running on first %d molecules in file only.", n)
    blocks = blocks[:n]

    # blocks = blocks[29500:]

    with ProcessPoolExecutor(max_workers=None) as executor, ProgressBar() as progress:
        # with ThreadPoolExecutor(initializer=silence_rdkit) as executor:
        task = progress.add_task("Submitting jobs: ", total=len(blocks))
        futures = []
        for i, block in enumerate(blocks):
            # replace the name of the molecule
            block = f"molblock_{i:07d}" + "\n" + block.split("\n", 1)[-1]
            futures.append(executor.submit(evaluate_one, block, pb=pb))
            progress.update(task, advance=1)

        results = []
        task = progress.add_task("Collecting jobs: ", total=len(futures))
        # for future in tqdm(as_completed(futures), total=len(futures), desc="Collecting jobs"):
        for future in as_completed(futures):
            try:
                results.append(future.result(timeout=timeout))
            except BrokenProcessPool as exception:
                logger.critical("BrokenProcessPool: %s", exception)
                results.append({"errorfree": 0, "error": str(exception).replace("\n", " ")})
            except Exception as exception:
                results.append({"errorfree": 0, "error": str(exception).replace("\n", " ")})
            progress.update(task, advance=1)

    results_df = pd.DataFrame(results).sort_values("name")
    if debug:
        print(results_df)
        return None
    output_file = output_file or input_file.with_suffix(".csv")
    results_df.to_csv(output_file, index=False)


if __name__ == "__main__":
    args = parse_arguments()
    setup_logging(level=args.logging)
    main(args.predicted, args.output, n=args.n, pb=args.nopb, debug=args.debug)
