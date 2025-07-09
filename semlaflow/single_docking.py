import argparse
import os
import sys
import numpy as np
from rdkit import RDLogger
import torch
from tqdm.auto import tqdm
from copy import deepcopy
sys.path.append('eval')
from util.docking_vina import VinaDockingTask
from multiprocessing import Pool
from functools import partial
from glob import glob
from rdkit import Chem


import os
import re
from rdkit.Chem import AllChem
from meeko import PDBQTMolecule

# from meeko import RDKitMolCreate
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Assuming the necessary functions and classes are correctly imported and set up
# You can adjust the imports based on your actual project structure

def main(mol, protein_filename, args, idx):

    # Adjust these paths as necessary

    try:

        vina_task = VinaDockingTask.from_generated_mol(deepcopy(mol),
                                                       protein_filename,
                                                       buffer=0.5)
        score_only_results = vina_task.run(mode='score_only', exhaustiveness=args.exhaustiveness)
        minimize_results = vina_task.run(mode='minimize', exhaustiveness=args.exhaustiveness,
                                         save_path=os.path.join(args.result_path,
                                                                f'{idx}_min_vina.pdbqt'))
        # minimize_results = vina_task.run(mode='minimize', exhaustiveness=args.exhaustiveness)

        vina_results = {
            'score_only': score_only_results,
            'minimize': minimize_results
        }
        if args.docking_mode == 'vina_full':
            # dock_results = vina_task.run(mode='dock', exhaustiveness=args.exhaustiveness)

            dock_results = vina_task.run(mode='dock', exhaustiveness=args.exhaustiveness,
                                         save_path=os.path.join(args.result_path, f'{idx}_dock_vina.pdbqt'))
            vina_results.update({
                'dock': dock_results,
            })

        # Add more conditions here if needed for other docking modes

        return vina_results

    except Exception as e:
        print("Error during docking: " + str(e))
        return None





def convert_pdbqt_to_sdf(ref_sdf_path, pdbqt_dir, output_sdf_path):
    # Load all reference ligands from the SDF file
    ref_ligands = list(Chem.SDMolSupplier(ref_sdf_path))

    # Collect and sort PDBQT files by numeric prefix
    pdbqt_files = [f for f in os.listdir(pdbqt_dir) if f.endswith('.pdbqt')]
    pdbqt_files = [f for f in pdbqt_files if re.search(r'^(\d+)', f)]  # Filter only valid files
    pdbqt_files.sort(key=lambda x: int(re.search(r'^(\d+)', x).group(1)))

    writer = Chem.SDWriter(output_sdf_path)

    for pdbqt_file in pdbqt_files:
        match = re.search(r'^(\d+)', pdbqt_file)
        if not match:
            continue
        index = int(match.group(1))
        if index >= len(ref_ligands):
            print(f"No reference ligand for {pdbqt_file}")
            continue

        input_file_path = os.path.join(pdbqt_dir, pdbqt_file)
        reflig = ref_ligands[index]

        pmol = PDBQTMolecule.from_file(input_file_path)
        for pose in pmol:
            output_rdmol = pose.export_rdkit_mol()
            output_rdmol_w_bond_order = AllChem.AssignBondOrdersFromTemplate(reflig, output_rdmol)
            # output_rdmol_w_bond_order = Chem.RemoveHs(output_rdmol_w_bond_order)
            writer.write(output_rdmol_w_bond_order)

    writer.close()
    print(f"Conversion complete. Output written to {output_sdf_path}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--protein_root', type=str, default='./../Data/CrossDocked')
    parser.add_argument('--input_sdf_file', type=str, default='./../Data/CrossDocked')
    parser.add_argument('--docking_mode', type=str, default='none',
                        choices=['none', 'qvina', 'vina', 'vina_full', 'vina_score'])
    parser.add_argument('--exhaustiveness', type=int, default=12)
    parser.add_argument('--result_path', type=str)

    # ligand_filename = 'ref_lig.sdf'

    protein_filename = 'protein.pdb'
    args = parser.parse_args()
    pdb_path = os.path.join(args.protein_root, protein_filename)
    sdf_reader = Chem.SDMolSupplier(args.input_sdf_file)
    for idx, ligand in enumerate(sdf_reader):


        results = main(ligand, pdb_path,args , idx)
        if results:
            print(results)