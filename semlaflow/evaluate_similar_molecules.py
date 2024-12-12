import os
from rdkit import Chem
from rdkit.Chem import QED, Descriptors, Crippen, Lipinski
from rdkit.Chem.FilterCatalog import *
from rdkit.Chem import rdMolDescriptors as rdMD
import numpy as np
from copy import deepcopy
from tqdm import tqdm
from semlaflow.util.metrics import Uniqueness,Novelty
import pandas as pd



def read_sdf(file_path):
    """Reads an SDF file and returns a list of molecules along with their names."""
    supplier = Chem.SDMolSupplier(file_path)
    molecules = []
    for mol in supplier:
        if mol is not None:
            name = mol.GetProp("_Name") if mol.HasProp("_Name") else "Unknown"
            molecules.append((name, mol))
    return molecules

def compute_sa_score(mol):
    try:
        sa_score = rdMD.CalcSPS(mol)
        return sa_score
    except Exception:
        return None

def get_logp(mol):
    return Crippen.MolLogP(mol)


def obey_lipinski(mol):
    mol = deepcopy(mol)
    Chem.SanitizeMol(mol)
    rule_1 = Descriptors.ExactMolWt(mol) < 500
    rule_2 = Lipinski.NumHDonors(mol) <= 5
    rule_3 = Lipinski.NumHAcceptors(mol) <= 10
    logp = get_logp(mol)
    rule_4 = (-2 <= logp <= 5)
    rule_5 = Chem.rdMolDescriptors.CalcNumRotatableBonds(mol) <= 10
    return [int(rule) for rule in [rule_1, rule_2, rule_3, rule_4, rule_5]]

def evaluate_metrics(generated_sdf, reference_sdf, batch_size=100):
    # Load molecules and group by reference names
    generated_mols = read_sdf(generated_sdf)
    reference_mols = read_sdf(reference_sdf)

    # Group generated molecules by their reference names
    grouped_mols = {}
    for name, mol in generated_mols:
        ref_name = name.split("_")[0]  # Assumes names are like 'reference_0', 'reference_1'
        if ref_name not in grouped_mols:
            grouped_mols[ref_name] = []
        grouped_mols[ref_name].append(mol)

    # Initialize metrics
    uniqueness_metric = Uniqueness()
    novelty_metric = Novelty([mol for _, mol in reference_mols])

    results = []

    # Process each group
    for ref_name, batch in tqdm(grouped_mols.items(), desc="Processing Groups"):
        uniqueness_metric.reset()
        uniqueness_metric.update(batch)
        novelty_metric.update(batch)

        qed_scores = []
        sa_scores = []
        normalized_sa_scores = []
        logp_scores = []
        lipinski_scores = []

        for mol in batch:
            if mol is not None:
                qed_score = QED.qed(mol)
                sa_score = compute_sa_score(mol)
                normalized_sa = sa_score / len([atom for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1])
                logp_score = get_logp(mol)
                lipinski_score = np.sum(obey_lipinski(mol))

                qed_scores.append(qed_score)
                sa_scores.append(sa_score)
                normalized_sa_scores.append(normalized_sa)
                logp_scores.append(logp_score)
                lipinski_scores.append(lipinski_score)

        group_results = {
            "Reference": ref_name,
            "Uniqueness": uniqueness_metric.compute(),
            "Novelty": novelty_metric.compute(),
            "Average QED": np.mean(qed_scores),
            "Average SA": np.mean(sa_scores),
            "Normalized SA": np.mean(normalized_sa_scores),
            "Average LogP": np.mean(logp_scores),
            "Average Lipinski Rules": np.mean(lipinski_scores),
        }
        results.append(group_results)

    # Compute mean and std for each metric
    results_df = pd.DataFrame(results)
    mean_metrics = results_df.iloc[:, 1:].mean()
    std_metrics = results_df.iloc[:, 1:].std()

    return results, mean_metrics, std_metrics


if __name__ == "__main__":
    generated_sdf_path = "./generated_molecules.sdf"  # Path to generated SDF file
    reference_sdf_path = "./reference_molecules.sdf"  # Path to reference SDF file

    results, mean_metrics, std_metrics = evaluate_metrics(generated_sdf_path, reference_sdf_path, batch_size=100)

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
    results_df.to_csv("evaluation_results.csv", index=False)
    print("\nResults saved to evaluation_results.csv")