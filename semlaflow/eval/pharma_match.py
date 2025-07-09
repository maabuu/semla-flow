import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from rdkit import Chem
from semlaflow.util.utils_pharmacophores import getPharamacophoreCoords  # Update if needed
from semlaflow.eval.utils import *
MATCH_DIST = 1.5  # Å threshold

def extract_ref_features(ref_mol):
    new_feats_dict, idxsDict, allCoords = getPharamacophoreCoords(ref_mol)
    ref_feats = []
    for fam, arr_list in idxsDict.items():
        for atom_ids in arr_list:
            coord = allCoords['Coords'][len(ref_feats)]
            ref_feats.append({
                "family": fam,
                "coord": np.array(coord),
                "atom_ids": np.atleast_1d(atom_ids).tolist()
            })
    return ref_feats

def extract_ref_features_combined(ref_mol):
    new_feats_dict, idxsDict, allCoords = getPharamacophoreCoords(ref_mol)
    ref_feats = []
    seen_coords_by_family = {}

    feat_counter = 0  # To track the index in allCoords['Coords']

    for fam, arr_list in idxsDict.items():
        # Only combine Donor and Acceptor
        if fam in ['Donor', 'Acceptor']:
            fam = 'DonorAcceptor'

        if fam not in seen_coords_by_family:
            seen_coords_by_family[fam] = set()

        for atom_ids in arr_list:
            coord = allCoords['Coords'][feat_counter]
            coord_key = tuple(np.round(coord, decimals=3))  # Round for float stability
            feat_counter += 1

            if coord_key not in seen_coords_by_family[fam]:
                seen_coords_by_family[fam].add(coord_key)
                ref_feats.append({
                    "family": fam,
                    "coord": np.array(coord),
                    "atom_ids": np.atleast_1d(atom_ids).tolist()
                })

    return ref_feats



from rdkit.Chem import AllChem

# def load_mols_from_sdf(path):
#     sup = Chem.SDMolSupplier(path, removeHs=False)
#     mols = []
#     for mol in sup:
#         if mol is not None:
#             mol = Chem.AddHs(mol)  # Add hydrogens
#             if mol.GetNumConformers() == 0:
#                 AllChem.EmbedMolecule(mol)  # Generate 3D coordinates if missing
#             else:
#                 conf = mol.GetConformer()
#                 if not conf.Is3D():
#                     AllChem.EmbedMolecule(mol)
#             mols.append(mol)
#     return mols



def analyze_pharmacophore_coverage_combined(ref_feats, gen_mols, thresh=1.5):
    match_counts = np.zeros(len(ref_feats), dtype=int)

    for gmol in gen_mols:
        gen_feats_dict, gen_idxs_dict, gen_all = getPharamacophoreCoords(gmol)
        gen_coords = np.array(gen_all['Coords'])
        gen_families = [
            'DonorAcceptor' if fam in ['Donor', 'Acceptor'] else fam
            for fam in gen_all['Family']
        ]

        for i, ref_feat in enumerate(ref_feats):
            ref_coord = ref_feat['coord']
            ref_family = ref_feat['family']
            matching_indices = [j for j, fam in enumerate(gen_families) if fam == ref_family]
            if matching_indices:
                dists = cdist([ref_coord], gen_coords[matching_indices])[0]
                if np.any(dists <= thresh):
                    match_counts[i] += 1

    df_feat = pd.DataFrame({
        'feature_index': range(len(ref_feats)),
        'family': [feat['family'] for feat in ref_feats],
        'match_count': match_counts,
        'match_fraction': match_counts / len(gen_mols)
    })
    return df_feat


def plot_combined_feature_bars(df_feat, out_png):
    families = df_feat['family'].unique()
    color_map = {fam: color for fam, color in zip(families, plt.cm.tab10.colors)}

    plt.figure(figsize=(14, 6))
    for fam in families:
        df_fam = df_feat[df_feat['family'] == fam]
        plt.bar(df_fam['feature_index'], df_fam['match_fraction'],
                color=color_map[fam], label=fam, edgecolor='k')

    plt.xlabel("Feature index")
    plt.ylabel("Match fraction")
    plt.ylim(0, 1)
    plt.title("Pharmacophore Match Fraction per Feature")
    plt.legend(title="Family")
    plt.tight_layout()
    plt.show()
    plt.savefig(out_png, dpi=150)
    plt.close()

# Example usage
if __name__ == "__main__":
    reference_sdf = r"C:\Users\ziv-admin\PycharmProjects\semla-flow\shepherd_data\NP_analogues_2500\orig\reference_molecule_2.sdf"
    generated_sdf = r"C:\Users\ziv-admin\PycharmProjects\semla-flow\predictions\NP_analogues_2500\ref_2\top10_Replacement_guidance.sdf"
    out_prefix = r"C:\Users\ziv-admin\PycharmProjects\semla-flow\predictions\NP_analogues_2500\ref_2\top10_Replacement_guidance"

    ref_mol = load_mols_from_sdf(reference_sdf)[0]
    gen_mols = load_mols_from_sdf(generated_sdf)

    ref_feats = extract_ref_features_combined(ref_mol)

    df_feat = analyze_pharmacophore_coverage_combined(ref_feats, gen_mols, thresh=MATCH_DIST)
    df_feat.to_csv(f"{out_prefix}_per_feature_coverage.csv", index=False)

    plot_combined_feature_bars(df_feat, f"{out_prefix}_combined_bars.png")

    # Example usage
    coords = [feat['coord'] for feat in ref_feats]  # from extract_ref_features
    match_fractions = df_feat['match_fraction'].tolist()  # from analyze_pharmacophore_coverage

    dummy_mol = create_dummy_molecule_with_bfactor(coords, match_fractions)

    Chem.MolToPDBFile(dummy_mol, f"{out_prefix}_ref_pharmacophore_dummy_with_properties.pdb")
    # Save to SDF
    # writer = Chem.SDWriter(f"{out_prefix}_ref_pharmacophore_dummy_with_properties.sdf")
    # writer.write(dummy_mol)
    # writer.close()

# spectrum b, blue_white_red, pharma, minimum=0, maximum=1
# label all, "%d%%" % (b * 100)
# set label_color, black







