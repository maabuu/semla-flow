import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, DataStructs
from itertools import combinations
import seaborn as sns
import os
from rdkit import Chem

def plot_similarity_opt(df_plot, title_suffix):
    sns.set(style="whitegrid")

    # Create the jointplot
    g = sns.jointplot(
        data=df_plot,
        x='sims_surf_target_relax_esp_aligned',
        y='sims_pharm_target_relax_esp_aligned',
        hue='method',
        kind='scatter',
        palette='Set2',
        alpha=0.4,
        height=7,
        marginal_kws=dict(common_norm=False, fill=True)
    )

    # Add KDE contours for each method
    for method in df_plot['method'].unique():
        subset = df_plot[df_plot['method'] == method]
        sns.kdeplot(
            data=subset,
            x='sims_surf_target_relax_esp_aligned',
            y='sims_pharm_target_relax_esp_aligned',
            ax=g.ax_joint,
            levels=5,
            linewidths=1,
            alpha=0.5
        )

    # Set axis labels and limits
    g.set_axis_labels("ESP Surface Similarity", "Pharmacophore Similarity", fontsize=12)
    g.ax_joint.set_xlim(0, 1)
    g.ax_joint.set_ylim(0, 1)

    # Add gridlines
    g.ax_joint.grid(True, linestyle='--', alpha=0.3)

    # Add marginal titles
    g.ax_marg_x.set_title("Distribution of ESP Similarity", fontsize=10)
    g.ax_marg_y.annotate(
        "Distribution of Pharmacophore Similarity",
        xy=(1.05, 0.5),
        xycoords='axes fraction',
        rotation=270,
        ha='center',
        va='center',
        fontsize=10
    )

    # Add main title
    g.fig.suptitle(f'ESP vs Pharmacophore Similarity {title_suffix}', fontsize=14)
    plt.subplots_adjust(top=0.92)

    plt.show()



import seaborn as sns
import matplotlib.pyplot as plt

def plot_graph_similarity_distribution(df, similarity_col='graph_similarities', method_col='method'):
    # Drop missing or invalid values
    df = df.dropna(subset=[similarity_col, method_col])

    # Set modern Seaborn style
    sns.set(style="whitegrid", context="talk")

    # Create the KDE plot
    plt.figure(figsize=(9, 6))
    sns.kdeplot(
        data=df,
        x=similarity_col,
        hue=method_col,
        common_norm=False,
        bw_adjust=0.5,
        linewidth=2.2,
        fill=False,
        palette="Set2"
    )

    # Customize plot aesthetics
    plt.title("NP2 - Graph Similarity Distribution", fontsize=16, weight='bold')
    plt.xlabel("Graph Similarity", fontsize=13)
    plt.ylabel("Density", fontsize=13)
    plt.xlim(0, 1)  # Ensure x-axis always ranges from 0 to 1
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()



def compute_diversity(df, molblock_col='molblocks', method_col='method', nbits=2048, radius=2):
    methods = df[method_col].unique()
    similarity_data = []
    for method in methods:
        df_method = df[df[method_col] == method]
        molblocks = df_method[molblock_col].dropna().astype(str)
        smiles_list = []
        for mb in molblocks:
            mol = Chem.MolFromMolBlock(mb, sanitize=False)
            if mol:
                smiles = Chem.MolToSmiles(mol)
                if smiles:
                    smiles_list.append(smiles)
        mols = [Chem.MolFromSmiles(smi) for smi in smiles_list if smi]
        fps = [rdMolDescriptors.GetMorganFingerprintAsBitVect(m, radius=radius, nBits=nbits) for m in mols if m]
        sims = [
            DataStructs.TanimotoSimilarity(fp1, fp2)
            for fp1, fp2 in combinations(fps, 2)
        ]
        similarity_data.extend([(method, sim) for sim in sims])
    sim_df = pd.DataFrame(similarity_data, columns=['method', 'similarity'])
    plt.figure(figsize=(8, 5))
    sns.kdeplot(data=sim_df, x='similarity', hue='method', common_norm=False, bw_adjust=0.5)
    plt.title("Pairwise Tanimoto Similarity Distributions")
    plt.xlabel("Tanimoto similarity")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.show()



def draw_top_mols_with_scores(top_mols, scores_df, filename="top_mols_annotated.png"):
    """
    Draws top molecules with annotated scores.

    Parameters:
    - top_mols: list of RDKit Mol objects
    - scores_df: DataFrame with columns ['SA_scores', 'sims_pharm_target_relax_esp_aligned', 'sims_surf_target_relax_esp_aligned']
                 Must be the same length/order as top_mols
    - filename: output image filename
    """
    for mol in top_mols:
        Chem.rdDepictor.Compute2DCoords(mol)

    drawer = rdMolDraw2D.MolDraw2DCairo(300 * len(top_mols), 300, 300, 300)
    opts = drawer.drawOptions()
    opts.useBWAtomPalette()
    opts.showAtomNumbers = False
    opts.addStereoAnnotation = True
    opts.explicitMethyl = True
    opts.bondLineWidth = 2.0
    opts.fixedBondLength = 25
    opts.minFontSize = 1  # must be int
    opts.legendFontSize = 18

    legends = [
        f"SA: {row['SA_scores']:.2f}\n\n\nPharm: {row['sims_pharm_target_relax_esp_aligned']:.2f}\n\n\nESP: {row['sims_surf_target_relax_esp_aligned']:.2f}"
        for i, row in scores_df.iterrows()
    ]

    drawer.DrawMolecules(top_mols, legends=legends)
    drawer.FinishDrawing()

    with open(filename, "wb") as f:
        f.write(drawer.GetDrawingText())


from rdkit import Chem, DataStructs
from rdkit.Chem import rdMolDescriptors
from itertools import combinations

def compute_diversity_hist(df,
                           molblock_col='molblocks',
                           method_col='method',
                           nbits=2048,
                           radius=2,
                           bins=50):
    """
    Computes pairwise Tanimoto similarity for each method and plots histograms.
    """
    sns.set(style="whitegrid", context="talk")
    methods = df[method_col].unique()
    plt.figure(figsize=(9, 6))

    for method in methods:
        df_m = df[df[method_col] == method]
        smiles_list = []
        for mb in df_m[molblock_col].dropna().astype(str):
            mol = Chem.MolFromMolBlock(mb, removeHs=False, sanitize=False)
            if mol:
                smi = Chem.MolToSmiles(mol)
                if smi:
                    smiles_list.append(smi)

        mols = [Chem.MolFromSmiles(smi) for smi in smiles_list]
        fps = [
            rdMolDescriptors.GetMorganFingerprintAsBitVect(m, radius=radius, nBits=nbits)
            for m in mols if m is not None
        ]

        sims = [
            DataStructs.TanimotoSimilarity(fp1, fp2)
            for fp1, fp2 in combinations(fps, 2)
        ]

        plt.hist(sims, bins=bins, alpha=0.5, density=True, label=method, edgecolor='black')

    plt.title("Pairwise Tanimoto Similarity Distributions", fontsize=16, weight='bold')
    plt.xlabel("Tanimoto Similarity", fontsize=13)
    plt.ylabel("Density", fontsize=13)
    plt.xlim(0, 1)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(title="Method", title_fontsize=12, fontsize=11)
    plt.tight_layout()
    plt.show()



def plot_box(df, col, title):
    sns.set(style="whitegrid", context="talk")

    plt.figure(figsize=(8, 6))
    sns.boxplot(
        data=df,
        x="method",
        y=col,
        palette="Set2",
        linewidth=2.0,     # Thicker box and whisker lines
        fliersize=6        # Larger outlier markers
    )

    plt.title(title, fontsize=16, weight='bold')
    plt.xlabel("")
    plt.ylabel(col.replace('_', ' ').title(), fontsize=13)

    plt.xticks(fontsize=14) # Smaller method labels

    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()




def save_top_n(df, method, n=50, output_dir="output"):
    """
    Save the top-n molecules for a given method into an SDF and CSV in the same order.

    Parameters:
    - df: DataFrame containing all methods
    - method: one of the method names in df["method"]
    - n: how many top molecules to save
    - output_dir: directory where the .sdf and .csv will be written
    """
    os.makedirs(output_dir, exist_ok=True)

    # filter & drop rows missing required columns
    df_m = (
        df[df["method"] == method]
        .dropna(subset=[
            # "molblocks",
            "molblocks_post_opt",
            "sims_pharm_target_relax_esp_aligned",
            "sims_surf_target_relax_esp_aligned"
        ]).copy()
    )

    # compute combined similarity
    df_m["total_similarity"] = (
        df_m["sims_pharm_target_relax_esp_aligned"] +
        df_m["sims_surf_target_relax_esp_aligned"]
    )

    # sort and take top-n
    topn = df_m.sort_values("total_similarity", ascending=False).head(n).reset_index(drop=True)

    # prepare output writers/paths
    sdf_path = os.path.join(output_dir, f"top{n}_{method}_orig.sdf")
    csv_path = os.path.join(output_dir, f"top{n}_{method}.csv")
    writer = Chem.SDWriter(sdf_path)

    smiles_list = []
    # iterate in DataFrame order
    for idx, row in topn.iterrows():
        mb = row["molblocks_post_opt"]
        mol = Chem.MolFromMolBlock(mb, removeHs=False)
        if mol:
            writer.write(mol)
            smiles_list.append(Chem.MolToSmiles(mol))
        else:
            smiles_list.append(None)
    writer.close()

    # add SMILES column (preserves same order as SDF)
    topn = topn.assign(SMILES=smiles_list)

    # select & save CSV
    cols = [
        "method", "SMILES", "SA_scores",
        "sims_pharm_target_relax_esp_aligned",
        "sims_surf_target_relax_esp_aligned",
        "total_similarity", "graph_similarities"
    ]
    topn[cols].to_csv(csv_path, index=False)

    print(f"Saved {n} molecules for '{method}' to:\n  SDF: {sdf_path}\n  CSV: {csv_path}")


# Load your CSVs

if __name__ == "__main__":

    df_shepherd = pd.read_csv(r"C:\Users\ziv-admin\PycharmProjects\semla-flow\shepherd_data\NP_analogues_2500\orig\sheperd_score_2_esp_align.csv")
    df_interpolate = pd.read_csv(r"C:\Users\ziv-admin\PycharmProjects\semla-flow\predictions\NP_analogues_2500\ref_2\interpolate_w_pharma_bonds_70_steps_36_80_esp_align.csv")
    df_merge = pd.read_csv(r"C:\Users\ziv-admin\PycharmProjects\semla-flow\predictions\NP_analogues_2500\ref_2\merge_w_pharma_esp_align_36_80.csv")

    # df_shepherd = pd.read_csv(r"C:\Users\ziv-admin\PycharmProjects\semla-flow\shepherd_data\Fragment_merging\samples_esp_align.csv")
    # df_interpolate = pd.read_csv(r"C:\Users\ziv-admin\PycharmProjects\semla-flow\predictions\Fragment_merging\merge\merge_random_50_89_esp_align.csv")
    # df_interpolate = pd.read_csv(r"C:\Users\ziv-admin\PycharmProjects\semla-flow\predictions\Fragment_merging\best_results\Replacement_guidance_docked\min_vina_docked_esp_align.csv")

    # df_merge = pd.read_csv(r"C:\Users\ziv-admin\PycharmProjects\semla-flow\predictions\Fragment_merging\merge\merge_sample_with_shepherd_pharma_esp_align.csv")
    # df_merge = pd.read_csv(r"C:\Users\ziv-admin\PycharmProjects\semla-flow\predictions\Fragment_merging\best_results\Replacement_guidance_docked_fixed_pharma\min_vina_docked_esp_align.csv")

    # Add method labels
    df_shepherd['method'] = 'ShEPhERD'
    df_interpolate['method'] = 'Integrate_interpolate'
    df_merge['method'] = "Replacement_guidance"

    # Combine
    df = pd.concat([df_shepherd, df_interpolate, df_merge], ignore_index=True)
    # df = pd.concat([df_shepherd, df_merge], ignore_index=True)

    # SA < 4.5 filter
    df_filtered = df[df['SA_scores'] < 4.5]

    # Print counts
    print(f"Total after SA < 4.5 filter: {len(df_filtered)}")
    for method in df_filtered['method'].unique():
        count = len(df_filtered[df_filtered['method'] == method])
        print(f"{method}: {count}")

    # Plotting function

    # Scatter plot: all
    # plot_similarity_opt(df, '(All Samples)')
    # Scatter plot: SA filtered
    # plot_similarity_opt(df_filtered, '(SA < 4.5)')

    # plot_graph_similarity_distribution(df_filtered)
    #
    #
    #
    # # Boxplot of SA scores
    # plot_box(df, "SA_scores", "SA Score by Method")
    # plot_box(df, "rmsds", "RMSD by Method")
    #
    # # compute_diversity(df)
    # compute_diversity_hist(df)
    for method in df_filtered['method'].unique():
        save_top_n(df_filtered, method, n=10, output_dir=r"C:\Users\ziv-admin\PycharmProjects\semla-flow\predictions\NP_analogues_2500\ref_2")
    #
