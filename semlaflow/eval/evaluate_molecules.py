"""Evaluate unconditionally generated molecules."""

import argparse
import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from functools import partial
from pathlib import Path
from typing import Dict, Union, Optional

import pandas as pd
from rdkit.Chem import QED, Crippen
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdMolDescriptors import CalcNumRotatableBonds
from rdkit.Chem.rdmolfiles import MolFromMolBlock, SDMolSupplier
from rdkit.Chem.rdmolops import AddHs, CombineMols, GetMolFrags, RemoveAllHs, RemoveHs, SanitizeMol
from rdkit.rdBase import LogToPythonLogger
from rich.logging import RichHandler
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TaskProgressColumn, TextColumn, TimeRemainingColumn, track

from tools import (
    compute_ghose_filter,
    compute_lipinski_score,
    compute_logp,
    compute_posebusters_validity,
    compute_qed_score,
    compute_sa_score,
    compute_smiles,
    compute_spacial_score,
    compute_weight,
    count_radicals,
    count_rings,
    compute_sucos,
    get_name,
    protect_from_segmentation_fault,
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
    help_line = "Previous results."
    parser.add_argument("--continue_file", type=Path, help=help_line, default=None)
    help_line = "Reference for similarity results."
    parser.add_argument("--reference_sdf", type=Path, help=help_line, default=None)
    help_line = "Disable PB checks."
    parser.add_argument("--nopb", action="store_false", help=help_line)  # by default PB checks on, if flag set then off
    help_line = "Logging level."
    parser.add_argument("--logging", "-l", type=int, help=help_line, default=logging.CRITICAL)
    help_line = "Enable debug mode."
    parser.add_argument("--debug", action="store_true", help=help_line)
    return parser.parse_args()


def evaluate_one(block: str, pb: bool = True, reference_mol: Optional[Mol] = None) -> Dict[str, Union[float, int, str]]:
    """Evaluate one molecule."""

    results = {}
    results["name"] = block.split("\n", 2)[0]
    results["loads"] = pd.NA
    results["sanitizes"] = pd.NA
    results["connected"] = pd.NA
    results["all_hydrogens"] = pd.NA
    results["no_radicals"] = pd.NA
    results["errorfree"] = pd.NA

    try:
        mol = MolFromMolBlock(block, sanitize=False, removeHs=False, strictParsing=False)
        assert mol is not None
        results["loads"] = True
    except Exception as e:
        results["error"] = str(e).replace("\n", " ")
        return results

    try:
        mol = MolFromMolBlock(block, sanitize=False, removeHs=False, strictParsing=False)
        mol.RemoveAllConformers()
        SanitizeMol(mol, catchErrors=False)
        results["sanitizes"] = True
        results["all_hydrogens"] = (AddHs(mol).GetNumAtoms() - mol.GetNumAtoms()) == 0
        results["no_radicals"] = count_radicals(mol) == 0
        results["connected"] = len(GetMolFrags(mol, asMols=False, sanitizeFrags=False)) == 1
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
        # results["spacial"] = compute_spacial_score(mol)
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
            protect_from_segmentation_fault(mol)
            posebuster_results = compute_posebusters_validity(mol)
            results = posebuster_results | results
        except Exception as e:
            results["error"] = str(e).replace("\n", " ")
            return results

        # SuCOS scoring if single reference provided
    if reference_mol is not None:
        try:
            results["sucos"] = compute_sucos(mol, reference_mol)
        except Exception as e:
            results["sucos_error"] = str(e)

    results["errorfree"] = True
    return results


def main(
    input_file: Path,
    output_file: Path,
    n: Optional[int]   = None,
    timeout: Optional[int] = None,
    debug: bool        = False,
    pb: bool           = True,
    continue_file: Optional[Path] = None,
    reference_sdf: Optional[Path]= None
) -> None:
    """Evaluate the molecules."""

    output_file = output_file or input_file.with_suffix(".csv")
    if Path(output_file).exists():
        raise FileExistsError(f"Output file {output_file} already exists.")

    mol_blocks = open(input_file).read().rstrip().rstrip("\n").rstrip("\n").rstrip("$$$$").split("$$$$\n")
    blocks = [(i, block) for i, block in enumerate(mol_blocks)]

    if continue_file is not None:
        already_complete = set(pd.read_csv(continue_file, low_memory=False)["name"].dropna().str.split("_").str[1].astype(int).unique())
        blocks = [block for block in blocks if block[0] not in already_complete]

    if n is not None:
        blocks = blocks[:n]

    # load single reference mol
    reference_mol: Optional[Mol] = None
    if reference_sdf:
        supplier = SDMolSupplier(str(reference_sdf))
        # take first non-None mol
        for m in supplier:
            if m is not None:
                reference_mol = m
                break

    with ProcessPoolExecutor(max_workers=None) as executor, ProgressBar() as progress:
        # with ThreadPoolExecutor(initializer=silence_rdkit) as executor:
        task = progress.add_task("Submitting jobs: ", total=len(blocks))
        task_block_ids = set()
        futures = []
        for i, block in blocks:
            # replace the name of the molecule
            task_block_ids.add(i)
            block = f"molblock_{i:07d}" + "\n" + block.split("\n", 1)[-1]
            futures.append(executor.submit(evaluate_one, block, pb=pb, reference_mol = reference_mol))
            progress.update(task, advance=1)

        results = []
        task = progress.add_task("Collecting jobs: ", total=len(futures))
        # for future in tqdm(as_completed(futures), total=len(futures), desc="Collecting jobs"):
        for future in as_completed(futures):
            try:
                result = future.result(timeout=timeout)
                results.append(result)
                task_block_ids.remove(int(result["name"].split("_")[1]))
            except BrokenProcessPool as exception:
                logger.critical("BrokenProcessPool: %s", exception)
                logger.critical("Unfinished IDs: %s", str(sorted(task_block_ids)))
                raise exception
            except Exception as exception:
                results.append({"errorfree": 0, "error": str(exception).replace("\n", " ")})
            progress.update(task, advance=1)

    results_df = pd.DataFrame(results).sort_values("name")
    if debug:
        print(results_df)
        return None
    results_df.to_csv(output_file, index=False)


if __name__ == "__main__":
    args = parse_arguments()
    setup_logging(level=args.logging)
    main(args.predicted, args.output, n=args.n, pb=args.nopb, debug=args.debug, continue_file=args.continue_file, reference_sdf = args.reference_sdf)
