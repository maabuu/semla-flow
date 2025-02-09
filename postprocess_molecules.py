"""Run MMFF energy minimization."""

import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from functools import partial
from pathlib import Path

from rdkit.Chem.MolStandardize.rdMolStandardize import CleanupParameters, LargestFragmentChooser
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdForceFieldHelpers import MMFFOptimizeMolecule
from rdkit.Chem.rdmolfiles import MolFromMolBlock, MolToMolBlock
from rdkit.Chem.rdmolops import AddHs, SanitizeMol
from rdkit.rdBase import LogToPythonLogger
from rich.logging import RichHandler
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TaskProgressColumn, TextColumn, TimeRemainingColumn, track

from tools import hydrate_radicals

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


params = CleanupParameters()
params.largestFragmentChooserCountHeavyAtomsOnly = True
params.preferOrganic = True
largest_fragment_chooser = LargestFragmentChooser()


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description="Evaluate molecules")
    help_line = "Path to SDF file containing predicted molecules."
    parser.add_argument("predicted", type=Path, help=help_line)
    help_line = "Output file."
    parser.add_argument("--output", "-o", type=Path, help=help_line, default=None)
    help_line = "Logging level."
    parser.add_argument("--logging", "-l", type=int, help=help_line, default=logging.CRITICAL)
    help_line = "Enable debug mode."
    parser.add_argument("--debug", action="store_true", help=help_line)
    return parser.parse_args()


def optimize_molecule(block: str, max_iters: int = 100000) -> Mol:
    """Minimize energy using MMFF energy minimization."""

    results = {}
    results["pass"] = False

    try:
        mol = MolFromMolBlock(block, sanitize=False, removeHs=False, strictParsing=False)
        assert mol is not None

        # pick largest fragment
        largest_fragment_chooser.chooseInPlace(mol)

        # sanitize molecule -  this includes adjusting Hydrogens
        SanitizeMol(mol)
        mol = hydrate_radicals(mol)

        # optimize conformation by minimizing energy
        mol = AddHs(mol, addCoords=True)
        result = MMFFOptimizeMolecule(mol, maxIters=max_iters)

        if result == -1:
            results["error"] = "Force field could not be set up."
            return results
        elif result == 1:
            logger.warning("More iterations required.")

        results["mol"] = MolToMolBlock(mol)

    except Exception as e:
        results["error"] = str(e).replace("\n", " ")
        return results

    results["pass"] = True
    return results


def main(input_file: Path, output_file: Path | None = None, debug: bool = False):
    """Run energy minimization on the molecules."""

    with open(input_file, "r") as reader:
        blocks = reader.read().rstrip().rstrip("\n").rstrip("\n").rstrip("$$$$").split("$$$$\n")
    total = len(blocks)

    if debug:
        logger.warning("Debug mode enabled.")
        total = 100

    with ThreadPoolExecutor() as executor, ProgressBar() as progress:
        futures = []
        task = progress.add_task("Submitting jobs: ", total=len(blocks))
        for i, block in enumerate(blocks):
            if debug and i == total:
                break
            futures.append(executor.submit(optimize_molecule, block))
            progress.update(task, advance=1)

        results = []
        task = progress.add_task("Collecting jobs: ", total=len(futures))
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except BrokenProcessPool as exception:
                raise exception
            except Exception as exception:
                results.append({"pass": False, "error": str(exception).replace("\n", " ")})
            progress.update(task, advance=1)

    # save optimized molecules
    output_file = output_file or input_file.with_stem(input_file.stem + "_optimized")

    output_blocks = "$$$$\n".join([result["mol"] for result in results if result["pass"]])
    with open(output_file, "w") as writer:
        writer.write(output_blocks)


if __name__ == "__main__":
    args = parse_arguments()
    setup_logging(level=args.logging)
    main(args.predicted, args.output, debug=args.debug)
