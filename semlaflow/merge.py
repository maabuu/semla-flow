"""
Script for generating molecules using a trained model and saving them.

Note that the script currently does not save the molecules in batches - all of the molecules are generated and then
all saved together in one Smol batch. If generating many molecules ensure you have enough memory to store them.
"""

import argparse
from pathlib import Path
from functools import partial

import torch
import lightning as L

import semlaflow.scriptutil as util
from semlaflow.util.molrepr import GeometricMolBatch
from semlaflow.models.fm import Integrator, MolecularCFM
from semlaflow.models.semla import EquiInvDynamics, SemlaGenerator

from semlaflow.data.datasets import GeometricDataset
from semlaflow.data.datamodules import GeometricInterpolantDM
from semlaflow.data.interpolate import GeometricInterpolant, GeometricNoiseSampler
from semlaflow.util.rdkit import write_mols_to_sdf
import numpy as np

# Default script arguments
DEFAULT_SAVE_FILE = "predictions.smol"
DEFAULT_DATASET_SPLIT = "test"
DEFAULT_N_MOLECULES = 5000
DEFAULT_BATCH_COST = 8192
DEFAULT_BUCKET_COST_SCALE = "linear"
DEFAULT_INTEGRATION_STEPS = 100
DEFAULT_DENOISE_STEPS = 50
DEFAULT_CAT_SAMPLING_NOISE_LEVEL = 1
DEFAULT_ODE_SAMPLING_STRATEGY = "log"

def load_model(args, vocab):
    checkpoint = torch.load(args.ckpt_path, map_location=torch.device('cpu'))
    hparams = checkpoint["hyper_parameters"]

    hparams["compile_model"] = False
    hparams["integration-steps"] = args.integration_steps
    hparams["sampling_strategy"] = args.ode_sampling_strategy

    n_bond_types = util.get_n_bond_types(hparams["integration-type-strategy"])

    # Set default arch to semla if nothing has been saved
    if hparams.get("architecture") is None:
        hparams["architecture"] = "semla"

    if hparams["architecture"] == "semla":
        dynamics = EquiInvDynamics(
            hparams["d_model"],
            hparams["d_message"],
            hparams["n_coord_sets"],
            hparams["n_layers"],
            n_attn_heads=hparams["n_attn_heads"],
            d_message_hidden=hparams["d_message_hidden"],
            d_edge=hparams["d_edge"],
            self_cond=hparams["self_cond"],
            coord_norm=hparams["coord_norm"],
        )
        egnn_gen = SemlaGenerator(
            hparams["d_model"],
            dynamics,
            vocab.size,
            hparams["n_atom_feats"],
            d_edge=hparams["d_edge"],
            n_edge_types=n_bond_types,
            self_cond=hparams["self_cond"],
            size_emb=hparams["size_emb"],
            max_atoms=hparams["max_atoms"]
        )

    elif hparams["architecture"] == "eqgat":
        from semlaflow.models.eqgat import EqgatGenerator

        egnn_gen = EqgatGenerator(
            hparams["d_model"],
            hparams["n_layers"],
            hparams["n_equi_feats"],
            vocab.size,
            hparams["n_atom_feats"],
            hparams["d_edge"],
            hparams["n_edge_types"]
        )

    elif hparams["architecture"] == "egnn":
        from semlaflow.models.egnn import VanillaEgnnGenerator

        n_layers = args.n_layers if hparams.get("n_layers") is None else hparams["n_layers"]
        if n_layers is None:
            raise ValueError("No hparam for n_layers was saved, use script arg to provide n_layers")

        egnn_gen = VanillaEgnnGenerator(
            hparams["d_model"],
            n_layers,
            vocab.size,
            hparams["n_atom_feats"],
            d_edge=hparams["d_edge"],
            n_edge_types=n_bond_types
        )

    else:
        raise ValueError(f"Unknown architecture hyperparameter.")

    type_mask_index = vocab.indices_from_tokens(["<MASK>"])[0] if hparams["train-type-interpolation"] == "mask" else None
    bond_mask_index = None

    integrator = Integrator(
        args.integration_steps,
        type_strategy=hparams["integration-type-strategy"],
        bond_strategy=hparams["integration-bond-strategy"],
        type_mask_index=type_mask_index,
        bond_mask_index=bond_mask_index,
        cat_noise_level=args.cat_sampling_noise_level
    )
    fm_model = MolecularCFM.load_from_checkpoint(
        args.ckpt_path,
        gen=egnn_gen,
        vocab=vocab,
        integrator=integrator,
        type_mask_index=type_mask_index,
        bond_mask_index=bond_mask_index,
        **hparams
    )
    return fm_model



def build_dm(args, hparams, vocab):
    if args.dataset == "qm9":
        coord_std = util.QM9_COORDS_STD_DEV
        bucket_limits = util.QM9_BUCKET_LIMITS

    elif args.dataset == "geom-drugs":
        coord_std = util.GEOM_COORDS_STD_DEV
        bucket_limits = util.GEOM_DRUGS_BUCKET_LIMITS

    else:
        raise ValueError(f"Unknown dataset {args.dataset}")

    n_bond_types = 5
    transform = partial(util.frag_transform, vocab=vocab, n_bonds=n_bond_types, coord_std=coord_std)

    if args.dataset_split == "train":
        dataset_path = Path(args.data_path) / "train.smol"
    elif args.dataset_split == "val":
        dataset_path = Path(args.data_path) / "val.smol"
    elif args.dataset_split == "test":
        dataset_path = Path(args.data_path) / "fragments.smol"

    dataset = GeometricDataset.load(dataset_path, transform=transform)
    dataset = dataset.sample(args.n_molecules, replacement=False, fixed_indices = True)

    type_mask_index = vocab.indices_from_tokens(["<MASK>"])[0] if hparams["val-type-interpolation"] == "mask" else None
    bond_mask_index = None

    if args.ode_sampling_strategy == "linear":
        time_points = np.linspace(0, 1, args.integration_steps + 1).tolist()
        time_points.reverse()

    elif args.ode_sampling_strategy == "log":
        time_points = (1 - np.geomspace(0.01, 1.0, args.integration_steps + 1)).tolist()


    fixed_time = time_points[args.denoise_steps]

    prior_sampler = GeometricNoiseSampler(
        vocab.size,
        n_bond_types,
        coord_noise="gaussian",
        type_noise=hparams["val-prior-type-noise"],
        bond_noise=hparams["val-prior-bond-noise"],
        scale_ot=False,
        zero_com=True,
        type_mask_index=type_mask_index,
        bond_mask_index=bond_mask_index
    )
    eval_interpolant = GeometricInterpolant(
        prior_sampler,
        coord_interpolation="linear",
        type_interpolation=hparams["val-type-interpolation"],
        bond_interpolation=hparams["val-bond-interpolation"],
        equivariant_ot=False,
        batch_ot=False,
        fixed_time = fixed_time,
    )
    dm = GeometricInterpolantDM(
        None,
        None,
        None,
        dataset,
        args.batch_cost,
        edit_interpolant=eval_interpolant,
        bucket_limits=bucket_limits,
        bucket_cost_scale=args.bucket_cost_scale,
        pad_to_bucket=False,
        merge=True
    )
    return dm, eval_interpolant


def dm_from_ckpt(args, vocab):
    checkpoint = torch.load(args.ckpt_path,map_location=torch.device('cpu'))
    hparams = checkpoint["hyper_parameters"]
    dm = build_dm(args, hparams, vocab)
    return dm


def generate_smol_mols(output, model):
    coords = output["coords"]
    atom_dists = output["atomics"]
    bond_dists = output["bonds"]
    charge_dists = output["charges"]
    masks = output["mask"]

    mols = model.builder.smol_from_tensors(
        coords,
        atom_dists,
        masks,
        bond_dists=bond_dists,
        charge_dists=charge_dists
    )
    return mols


def save_predictions(args, molecules, raw_outputs, model):
    save_path = Path(args.save_dir) / args.save_file
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if args.save_file.endswith(".sdf"):
        return write_mols_to_sdf(molecules, str(save_path))

    # Generate GeometricMols and then combine into one GeometricMolBatch
    mol_lists = [generate_smol_mols(output, model) for output in raw_outputs]
    mols = [mol for mol_list in mol_lists for mol in mol_list]
    batch = GeometricMolBatch.from_list(mols)

    batch_bytes = batch.to_bytes()
    save_path.write_bytes(batch_bytes)


def main(args):
    print(f"Running prediction script for {args.n_molecules} molecules...")
    print(f"Using model stored at {args.ckpt_path}")

    L.seed_everything(12345)
    util.disable_lib_stdout()
    # util.configure_fs()

    print("Building model vocab...")
    vocab = util.build_vocab()
    print("Vocab complete.")

    print("Loading datamodule...")
    dm, merge_interpolant = dm_from_ckpt(args, vocab)
    print("Datamodule complete.")

    print(f"Loading model...")
    model = load_model(args, vocab)
    print("Model complete.")


    # print("Initialising metrics...")
    # metrics, _ = util.init_metrics(args.data_path, model)
    # print("Metrics complete.")

    print("Running generation...")
    molecules, raw_outputs = util.merge_fragments( model, dm, steps= args.integration_steps,  merge_interpolant = merge_interpolant, strategy=args.ode_sampling_strategy)
    print("Generation complete.")

    print(f"Saving predictions to {args.save_dir}/{args.save_file}")
    save_predictions(args, molecules, raw_outputs, model)
    print("Complete.")

    print("Calculating generative metrics...")
    # results = util.calc_metrics_(molecules, metrics)
    # util.print_results(results)
    # print("Generation script complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--ckpt_path", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--save_file", type=str, default=DEFAULT_SAVE_FILE)

    parser.add_argument("--batch_cost", type=int, default=DEFAULT_BATCH_COST)
    parser.add_argument("--dataset_split", type=str, default=DEFAULT_DATASET_SPLIT)
    parser.add_argument("--n_molecules", type=int, default=DEFAULT_N_MOLECULES)
    parser.add_argument("--integration_steps", type=int, default=DEFAULT_INTEGRATION_STEPS)
    parser.add_argument("--denoise_steps", type=int, default=DEFAULT_DENOISE_STEPS)
    parser.add_argument("--sigma", type=float, default=0.1)

    parser.add_argument("--cat_sampling_noise_level", type=int, default=DEFAULT_CAT_SAMPLING_NOISE_LEVEL)
    parser.add_argument("--ode_sampling_strategy", type=str, default=DEFAULT_ODE_SAMPLING_STRATEGY)

    parser.add_argument("--bucket_cost_scale", type=str, default=DEFAULT_BUCKET_COST_SCALE)
    import torch.multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    args = parser.parse_args()
    main(args)
