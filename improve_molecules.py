"""Run MMFF energy minimization."""

import argparse
import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

from rdkit.Chem.MolStandardize.rdMolStandardize import LargestFragmentChooser
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdForceFieldHelpers import MMFFOptimizeMolecule
from rdkit.Chem.rdmolfiles import MolFromMolBlock, MolToMolBlock
from rdkit.Chem.rdmolops import AddHs
from tqdm import tqdm

from tools import silence_rdkit

logger = logging.getLogger(__name__)
largest_fragment_chooser = LargestFragmentChooser()


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description="Evaluate molecules")
    help_line = "Path to SDF file containing predicted molecules."
    parser.add_argument("predicted", type=Path, help=help_line)
    help_line = "Output file."
    parser.add_argument("--output", "-o", type=Path, help=help_line, default=None)
    help_line = "Enable debug mode."
    parser.add_argument("--debug", action="store_true", help=help_line)
    return parser.parse_args()


def optimize_molecule(block: str, max_iters: int = 100000) -> Mol:
    """Minimize energy using MMFF energy minimization."""

    results = {}
    results["pass"] = False

    try:
        # sanitize molecule - this includes adjusting Hydrogens
        mol = MolFromMolBlock(block, sanitize=True, removeHs=False, strictParsing=True)
        assert mol is not None

        # pick largest fragment
        largest_mol = largest_fragment_chooser.choose(mol)

        # add hydrogens and optimize
        mol = AddHs(largest_mol, addCoords=True)
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

    blocks = open(input_file).read().strip().strip("\n").strip("$").split("$$$$\n")
    total = len(blocks)

    if debug:
        logger.warning("Debug mode enabled.")
        total = 100

    # with ProcessPoolExecutor(initializer=silence_rdkit) as executor:
    with ThreadPoolExecutor(initializer=silence_rdkit) as executor:
        futures = []
        for i, block in enumerate(tqdm(blocks, total=total, desc="Submitting jobs")):
            if debug and i == total:
                break
            futures.append(executor.submit(optimize_molecule, block))

        results = []
        for future in tqdm(as_completed(futures), total=len(futures), desc="Collecting jobs"):
            try:
                results.append(future.result())
            except BrokenProcessPool as exception:
                raise exception
            except Exception as exception:
                results.append({"pass": False, "error": str(exception).replace("\n", " ")})

    # save optimized molecules
    output_file = output_file or input_file.with_stem(input_file.stem + "_optimized")

    with open(output_file, "w") as writer:
        for result in results:
            if result["pass"]:
                writer.write(result["mol"])
                writer.write("$$$$\n")


if __name__ == "__main__":
    args = parse_arguments()
    main(args.predicted, args.output, debug=args.debug)
