import os
import sys
from copy import deepcopy

import numpy as np
import pandas as pd
from posebusters import PoseBusters
from rdkit import Chem
from rdkit.Chem import QED, Crippen, Descriptors, Lipinski
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdmolfiles import SDMolSupplier
from rdkit.Chem.rdmolops import SanitizeMol
from rdkit.Chem.SpacialScore import SPS
from tqdm import tqdm
from rdkit import RDConfig
from rdkit.Chem import AllChem, rdShapeHelpers
from rdkit.Chem.FeatMaps import FeatMaps
from rdkit.Chem import SDWriter


from semlaflow.util.metrics import Novelty, Uniqueness

try:
    # if rdkit was installed from pip
    # https://greglandrum.github.io/rdkit-blog/posts/2023-12-01-using_sascore_and_npscore.html
    from rdkit.Contrib.SA_Score import sascorer
except ImportError:
    # if rdkit was installed from conda
    sys.path.append(
        os.path.join(os.environ["CONDA_PREFIX"], "share", "RDKit", "Contrib")
    )
    from SA_Score import sascorer

buster = PoseBusters("mol")


def read_sdf(file_path: str) -> list[Mol]:
    """Reads an SDF file and returns a list of molecules along with their names."""
    supplier = SDMolSupplier(file_path)
    molecules = []
    for mol in supplier:
        if mol is not None:
            name = mol.GetProp("_Name") if mol.HasProp("_Name") else "Unknown"
            molecules.append((name, mol))
    return molecules


def compute_chemical_and_physical_validity(mol: Mol):
    pb_results: dict[str, bool] = buster.bust(mol).iloc[0].to_dict()

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
    pb_results |= {
        "connected": all(pb_results[check] for check in check_connected),
        "chemical": all(pb_results[check] for check in checks_chemical),
        "physical": all(pb_results[check] for check in checks_physical),
    }

    return pb_results

fdef = AllChem.BuildFeatureFactory(os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef'))
fmParams = {}
for k in fdef.GetFeatureFamilies():
    fparams = FeatMaps.FeatMapParams()
    fmParams[k] = fparams

keep = ('Donor', 'Acceptor', 'NegIonizable', 'PosIonizable', 'ZnBinder',
        'Aromatic', 'Hydrophobe', 'LumpedHydrophobe')
def get_FeatureMapScore(small_m, large_m, score_mode=FeatMaps.FeatMapScoreMode.Best):
    featLists = []
    for m in [small_m, large_m]:
        rawFeats = fdef.GetFeaturesForMol(m)
        # filter that list down to only include the ones we're intereted in
        featLists.append([f for f in rawFeats if f.GetFamily() in keep])
    fms = [FeatMaps.FeatMap(feats=x, weights=[1] * len(x), params=fmParams) for x in featLists]
    fms[0].scoreMode = score_mode
    fm_score = fms[0].ScoreFeats(featLists[1]) / min(fms[0].GetNumFeatures(), len(featLists[1]))
    return fm_score
def compute_SuCOS_score(mol, ref_mol):
    """
    Computes the SuCOS score between a generated molecule and its reference molecule.
    SuCOS = 0.5 * FeatureMapScore + 0.5 * (1 - ProtrudeDist)
    """
    if mol is None or ref_mol is None:
        return float("nan")
    try:
        # Feature Map Score
        fm_score = get_FeatureMapScore(ref_mol, mol)

        # Protrude Distance
        protrude_dist = rdShapeHelpers.ShapeProtrudeDist(ref_mol, mol, allowReordering=False)

        # SuCOS Score
        SuCOS_score = 0.5 * fm_score + 0.5 * (1 - protrude_dist)

        return SuCOS_score
    except Exception as e:
        print(f"Error calculating SuCOS: {e}")
        return float("nan")
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


def compute_logp(mol: Mol) -> float:
    """Compute the logP of a molecule using the Crippen method.
    RDKit reference: https://www.rdkit.org/docs/source/rdkit.Chem.Crippen.html
    Paper reference: https://pubs.acs.org/doi/10.1021/ci990307l
    """
    try:
        return float(Crippen.MolLogP(mol))
    except Exception:
        return float("nan")


def obey_lipinski(mol: Mol) -> tuple[int, int, int, int, int]:
    """Compute the Lipinski rule of 5 for a molecule.
    RDKit reference: https://www.rdkit.org/docs/GettingStartedInPython.html#lipinski-rule-of-5
    Paper reference: https://www.sciencedirect.com/science/article/pii/S0169409X00001290
    """
    mol = deepcopy(mol)
    SanitizeMol(mol)
    rule_1 = Descriptors.ExactMolWt(mol) < 500
    rule_2 = Lipinski.NumHDonors(mol) <= 5
    rule_3 = Lipinski.NumHAcceptors(mol) <= 10
    logp = compute_logp(mol)
    rule_4 = -2 <= logp <= 5
    rule_5 = Chem.rdMolDescriptors.CalcNumRotatableBonds(mol) <= 10
    return [int(rule) for rule in [rule_1, rule_2, rule_3, rule_4, rule_5]]


def evaluate_metrics(generated_sdf, reference_sdf, batch_size=100, results_path = None):
    # Load molecules and group by reference names
    generated_mols = read_sdf(generated_sdf)
    reference_mols = SDMolSupplier(reference_sdf)

    # Group generated molecules by their reference names
    grouped_mols = {}
    for name, mol in generated_mols:
        ref_name = name.split("_")[
            1
        ]  # Assumes names are like 'reference_0', 'reference_1'
        if ref_name not in grouped_mols:
            grouped_mols[ref_name] = []
        grouped_mols[ref_name].append(mol)

    # Check for missing groups in range 0-999
    expected_groups = {str(i) for i in range(1000)}  # Set of expected group names
    existing_groups = set(grouped_mols.keys())  # Set of existing group names
    missing_groups = expected_groups - existing_groups
    if missing_groups:
        print(f"Warning: The following groups are missing (no molecules): {sorted(missing_groups)}")
        print(len(missing_groups))

        # Save each group as a separate SDF file
    if results_path:
        os.makedirs(results_path, exist_ok=True)  # Ensure the results directory exists

    for ref_name in expected_groups:  # Iterate over all expected groups
        group = grouped_mols.get(ref_name, [])  # Retrieve group or empty list if missing

        # Save the group as a separate SDF file
        sdf_path = os.path.join(results_path, f"group_{ref_name}.sdf")
        with SDWriter(sdf_path) as writer:
            for mol in group:
                writer.write(mol)

    # Initialize metrics
    uniqueness_metric = Uniqueness()


    results = []

    # Process each group
    for ref_name, batch in tqdm(grouped_mols.items(), desc="Processing Groups"):
        uniqueness_metric.reset()
        uniqueness_metric.update(batch)

        qed_scores = []
        sa_scores = []
        normalized_sa_scores = []
        logp_scores = []
        lipinski_scores = []
        chemical_validities = []
        physical_validities = []
        SuCOS_scores = []

        ref_mol = reference_mols[int(ref_name)]
        novelty_metric = Novelty([ref_mol])
        novelty_metric.update(batch)
        for mol in batch:
            if mol is not None:
                validity = compute_chemical_and_physical_validity(mol)
                qed_score = QED.qed(mol)
                sa_score = compute_sa_score(mol)
                normalized_sa = sa_score / len(
                    [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1]
                )
                logp_score = compute_logp(mol)
                lipinski_score = np.sum(obey_lipinski(mol))
                SuCOS_score = compute_SuCOS_score(mol, ref_mol)

                qed_scores.append(qed_score)
                sa_scores.append(sa_score)
                normalized_sa_scores.append(normalized_sa)
                logp_scores.append(logp_score)
                lipinski_scores.append(lipinski_score)
                chemical_validities.append(validity["chemical"])
                physical_validities.append(validity["physical"])
                SuCOS_scores.append(SuCOS_score)

        group_results = {
            "Reference": ref_name,
            "Uniqueness": float(uniqueness_metric.compute()),  # Convert tensor to float
            "Novelty": float(novelty_metric.compute()),      # If enabled, convert as well
            "Average QED": float(np.mean(qed_scores)),
            "Average SA": float(np.mean(sa_scores)),
            "Normalized SA": float(np.mean(normalized_sa_scores)),
            "Average LogP": float(np.mean(logp_scores)),
            "Average Lipinski Rules": float(np.mean(lipinski_scores)),
            "Chemical Validity (%)": float(np.mean(chemical_validities)) * 100,
            "Physical Validity (%)": float(np.mean(physical_validities)) * 100,
            "Average SuCOS": float(np.mean(SuCOS_scores)),
        }
        results.append(group_results)

    # Compute mean and std for each metric
    results_df = pd.DataFrame(results)
    if results_path is not None:
        base_name = os.path.splitext(os.path.basename(generated_sdf))[0]
        results_df.to_csv(results_path+ f"/results_table{base_name}.csv", index=False)
    mean_metrics = results_df.iloc[:, 1:].mean()
    std_metrics = results_df.iloc[:, 1:].std()

    return results, mean_metrics, std_metrics


if __name__ == "__main__":
    generated_sdf_path = "../predictions/predictions_d72_s0.10.sdf"  # Path to generated SDF file
    reference_sdf_path = r"C:\Users\ziv-admin\PycharmProjects\semla-flow\geom_data\smol\test_first_1000_rdkit.sdf"  # Path to reference SDF file
    results_path = '../predictions/72_steps'

    results, mean_metrics, std_metrics = evaluate_metrics(
        generated_sdf_path, reference_sdf_path, batch_size=100, results_path=results_path
    )

    # Print results for each group
    print("\nEvaluation Results (per group):")
    for group in results:
        print(group)

    # Print mean and standard deviation of all metrics
    print("\nMean Metrics Across All Groups:")
    print(mean_metrics)

    print("\nStandard Deviation of Metrics Across All Groups:")
    print(std_metrics)

    # Save results to a CSV file
    results_df = pd.DataFrame(results)
    base_name = os.path.splitext(os.path.basename(generated_sdf_path))[0]
    results_df.to_csv(results_path + f"/evaluation_results{base_name}.csv", index=False)
    print("\nResults saved to evaluation_results.csv")