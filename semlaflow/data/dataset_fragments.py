import os
import numpy as np
import pandas as pd
import pickle
import torch

from rdkit import Chem
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from semlaflow.data import const


from pdb import set_trace

from rdkit import Chem, Geometry


def create_conformer(coords):
    conformer = Chem.Conformer()
    for i, (x, y, z) in enumerate(coords):
        conformer.SetAtomPosition(i, Geometry.Point3D(x, y, z))
    return conformer


def build_molecules(one_hot, x, node_mask, is_geom, margins=const.MARGINS_EDM):
    molecules = []
    for i in range(len(one_hot)):
        mask = node_mask[i].squeeze() == 1
        atom_types = one_hot[i][mask].argmax(dim=1).detach().cpu()
        positions = x[i][mask].detach().cpu()
        mol = build_molecule(positions, atom_types, is_geom, margins=margins)
        molecules.append(mol)

    return molecules


def build_molecule(positions, atom_types, is_geom, margins=const.MARGINS_EDM):
    idx2atom = const.GEOM_IDX2ATOM if is_geom else const.IDX2ATOM
    X, A, E = build_xae_molecule(positions, atom_types, is_geom=is_geom, margins=margins)
    mol = Chem.RWMol()
    for atom in X:
        a = Chem.Atom(idx2atom[atom.item()])
        mol.AddAtom(a)

    all_bonds = torch.nonzero(A)
    for bond in all_bonds:
        mol.AddBond(bond[0].item(), bond[1].item(), const.BOND_DICT[E[bond[0], bond[1]].item()])

    mol.AddConformer(create_conformer(positions.detach().cpu().numpy().astype(np.float64)))
    return mol


def build_xae_molecule(positions, atom_types, is_geom, margins=const.MARGINS_EDM):
    """ Returns a triplet (X, A, E): atom_types, adjacency matrix, edge_types
        args:
        positions: N x 3  (already masked to keep final number nodes)
        atom_types: N
        returns:
        X: N         (int)
        A: N x N     (bool) (binary adjacency matrix)
        E: N x N     (int)  (bond type, 0 if no bond) such that A = E.bool()
    """
    n = positions.shape[0]
    X = atom_types
    A = torch.zeros((n, n), dtype=torch.bool)
    E = torch.zeros((n, n), dtype=torch.int)

    idx2atom = const.GEOM_IDX2ATOM if is_geom else const.IDX2ATOM

    pos = positions.unsqueeze(0)
    dists = torch.cdist(pos, pos, p=2).squeeze(0)
    for i in range(n):
        for j in range(i):

            pair = sorted([atom_types[i], atom_types[j]])
            order = get_bond_order(idx2atom[pair[0].item()], idx2atom[pair[1].item()], dists[i, j], margins=margins)

            # TODO: a batched version of get_bond_order to avoid the for loop
            if order > 0:
                # Warning: the graph should be DIRECTED
                A[i, j] = 1
                E[i, j] = order

    return X, A, E


def get_bond_order(atom1, atom2, distance, check_exists=True, margins=const.MARGINS_EDM):
    distance = 100 * distance  # We change the metric

    # Check exists for large molecules where some atom pairs do not have a
    # typical bond length.
    if check_exists:
        if atom1 not in const.BONDS_1:
            return 0
        if atom2 not in const.BONDS_1[atom1]:
            return 0

    # margin1, margin2 and margin3 have been tuned to maximize the stability of the QM9 true samples
    if distance < const.BONDS_1[atom1][atom2] + margins[0]:

        # Check if atoms in bonds2 dictionary.
        if atom1 in const.BONDS_2 and atom2 in const.BONDS_2[atom1]:
            thr_bond2 = const.BONDS_2[atom1][atom2] + margins[1]
            if distance < thr_bond2:
                if atom1 in const.BONDS_3 and atom2 in const.BONDS_3[atom1]:
                    thr_bond3 = const.BONDS_3[atom1][atom2] + margins[2]
                    if distance < thr_bond3:
                        return 3  # Triple
                return 2  # Double
        return 1  # Single
    return 0  # No bond
def read_sdf(sdf_path):
    with Chem.SDMolSupplier(sdf_path, sanitize=False) as supplier:
        for molecule in supplier:
            yield molecule


def get_one_hot(atom, atoms_dict):
    one_hot = np.zeros(len(atoms_dict))
    one_hot[atoms_dict[atom]] = 1
    return one_hot


def parse_molecule(mol, is_geom):
    one_hot = []
    charges = []
    atom2idx = const.GEOM_ATOM2IDX if is_geom else const.ATOM2IDX
    charges_dict = const.GEOM_CHARGES if is_geom else const.CHARGES
    for atom in mol.GetAtoms():
        one_hot.append(get_one_hot(atom.GetSymbol(), atom2idx))
        charges.append(charges_dict[atom.GetSymbol()])
    positions = mol.GetConformer().GetPositions()
    return positions, np.array(one_hot), np.array(charges)


class ZincDataset(Dataset):
    def __init__(self, data_path, prefix, device):
        dataset_path = os.path.join(data_path, f'{prefix}.pt')
        if os.path.exists(dataset_path):
            self.data = torch.load(dataset_path, map_location=device)
        else:
            print(f'Preprocessing dataset with prefix {prefix}')
            self.data = ZincDataset.preprocess(data_path, prefix, device)
            torch.save(self.data, dataset_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    @staticmethod
    def preprocess(data_path, prefix, device):
        data = []
        table_path = os.path.join(data_path, f'{prefix}_table.csv')
        fragments_path = os.path.join(data_path, f'{prefix}_frag.sdf')
        linkers_path = os.path.join(data_path, f'{prefix}_link.sdf')

        is_geom = ('geom' in prefix) or ('MOAD' in prefix)
        is_multifrag = 'multifrag' in prefix

        table = pd.read_csv(table_path)
        generator = tqdm(zip(table.iterrows(), read_sdf(fragments_path), read_sdf(linkers_path)), total=len(table))
        for (_, row), fragments, linker in generator:
            uuid = row['uuid']
            name = row['molecule']
            frag_pos, frag_one_hot, frag_charges = parse_molecule(fragments, is_geom=is_geom)
            link_pos, link_one_hot, link_charges = parse_molecule(linker, is_geom=is_geom)

            positions = np.concatenate([frag_pos, link_pos], axis=0)
            one_hot = np.concatenate([frag_one_hot, link_one_hot], axis=0)
            charges = np.concatenate([frag_charges, link_charges], axis=0)
            anchors = np.zeros_like(charges)

            if is_multifrag:
                for anchor_idx in map(int, row['anchors'].split('-')):
                    anchors[anchor_idx] = 1
            else:
                anchors[row['anchor_1']] = 1
                anchors[row['anchor_2']] = 1
            fragment_mask = np.concatenate([np.ones_like(frag_charges), np.zeros_like(link_charges)])
            linker_mask = np.concatenate([np.zeros_like(frag_charges), np.ones_like(link_charges)])

            data.append({
                'uuid': uuid,
                'name': name,
                'positions': torch.tensor(positions, dtype=const.TORCH_FLOAT, device=device),
                'one_hot': torch.tensor(one_hot, dtype=const.TORCH_FLOAT, device=device),
                'charges': torch.tensor(charges, dtype=const.TORCH_FLOAT, device=device),
                'anchors': torch.tensor(anchors, dtype=const.TORCH_FLOAT, device=device),
                'fragment_mask': torch.tensor(fragment_mask, dtype=const.TORCH_FLOAT, device=device),
                'linker_mask': torch.tensor(linker_mask, dtype=const.TORCH_FLOAT, device=device),
                'num_atoms': len(positions),
            })

        return data


class MOADDataset(Dataset):
    def __init__(self, data=None, data_path=None, prefix=None, device=None):
        assert (data is not None) or all(x is not None for x in (data_path, prefix, device))
        if data is not None:
            self.data = data
            return

        if '.' in prefix:
            prefix, pocket_mode = prefix.split('.')
        else:
            parts = prefix.split('_')
            prefix = '_'.join(parts[:-1])
            pocket_mode = parts[-1]

        dataset_path = os.path.join(data_path, f'{prefix}_{pocket_mode}.pt')
        if os.path.exists(dataset_path):
            self.data = torch.load(dataset_path, map_location=device)
        else:
            print(f'Preprocessing dataset with prefix {prefix}')
            self.data = self.preprocess(data_path, prefix, pocket_mode, device)
            torch.save(self.data, dataset_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    @staticmethod
    def preprocess(data_path, prefix, pocket_mode, device):
        data = []
        table_path = os.path.join(data_path, f'{prefix}_table.csv')
        fragments_path = os.path.join(data_path, f'{prefix}_frag.sdf')
        linkers_path = os.path.join(data_path, f'{prefix}_link.sdf')
        pockets_path = os.path.join(data_path, f'{prefix}_pockets.pkl')

        is_geom = True
        is_multifrag = 'multifrag' in prefix

        with open(pockets_path, 'rb') as f:
            pockets = pickle.load(f)

        table = pd.read_csv(table_path)
        generator = tqdm(
            zip(table.iterrows(), read_sdf(fragments_path), read_sdf(linkers_path), pockets),
            total=len(table)
        )
        for (_, row), fragments, linker, pocket_data in generator:
            pdb = row['molecule_name'].split('_')[0]
            if pdb in {
                '5ou2', '5ou3', '6hay',
                '5mo8', '5mo5', '5mo7', '5ctp', '5cu2', '5cu4', '5mmr', '5mmf',
                '5moe', '3iw7', '4i9n', '3fi2', '3fi3',
            }:
                print(f'Skipping pdb={pdb}')
                continue

            uuid = row['uuid']
            name = row['molecule']
            frag_pos, frag_one_hot, frag_charges = parse_molecule(fragments, is_geom=is_geom)
            link_pos, link_one_hot, link_charges = parse_molecule(linker, is_geom=is_geom)

            # Parsing pocket data
            pocket_pos = pocket_data[f'{pocket_mode}_coord']
            pocket_one_hot = []
            pocket_charges = []
            for atom_type in pocket_data[f'{pocket_mode}_types']:
                pocket_one_hot.append(get_one_hot(atom_type, const.GEOM_ATOM2IDX))
                pocket_charges.append(const.GEOM_CHARGES[atom_type])
            pocket_one_hot = np.array(pocket_one_hot)
            pocket_charges = np.array(pocket_charges)

            positions = np.concatenate([frag_pos, pocket_pos, link_pos], axis=0)
            one_hot = np.concatenate([frag_one_hot, pocket_one_hot, link_one_hot], axis=0)
            charges = np.concatenate([frag_charges, pocket_charges, link_charges], axis=0)
            anchors = np.zeros_like(charges)

            if is_multifrag:
                for anchor_idx in map(int, row['anchors'].split('-')):
                    anchors[anchor_idx] = 1
            else:
                anchors[row['anchor_1']] = 1
                anchors[row['anchor_2']] = 1

            fragment_only_mask = np.concatenate([
                np.ones_like(frag_charges),
                np.zeros_like(pocket_charges),
                np.zeros_like(link_charges)
            ])
            pocket_mask = np.concatenate([
                np.zeros_like(frag_charges),
                np.ones_like(pocket_charges),
                np.zeros_like(link_charges)
            ])
            linker_mask = np.concatenate([
                np.zeros_like(frag_charges),
                np.zeros_like(pocket_charges),
                np.ones_like(link_charges)
            ])
            fragment_mask = np.concatenate([
                np.ones_like(frag_charges),
                np.ones_like(pocket_charges),
                np.zeros_like(link_charges)
            ])

            data.append({
                'uuid': uuid,
                'name': name,
                'positions': torch.tensor(positions, dtype=const.TORCH_FLOAT, device=device),
                'one_hot': torch.tensor(one_hot, dtype=const.TORCH_FLOAT, device=device),
                'charges': torch.tensor(charges, dtype=const.TORCH_FLOAT, device=device),
                'anchors': torch.tensor(anchors, dtype=const.TORCH_FLOAT, device=device),
                'fragment_only_mask': torch.tensor(fragment_only_mask, dtype=const.TORCH_FLOAT, device=device),
                'pocket_mask': torch.tensor(pocket_mask, dtype=const.TORCH_FLOAT, device=device),
                'fragment_mask': torch.tensor(fragment_mask, dtype=const.TORCH_FLOAT, device=device),
                'linker_mask': torch.tensor(linker_mask, dtype=const.TORCH_FLOAT, device=device),
                'num_atoms': len(positions),
            })

        return data


class OptimisedMOADDataset(MOADDataset):
    # TODO: finish testing

    def __len__(self):
        return len(self.data['fragmentation_level_data'])

    def __getitem__(self, item):
        fragmentation_level_data = self.data['fragmentation_level_data'][item]
        protein_level_data = self.data['protein_level_data'][fragmentation_level_data['name']]
        return {
            **fragmentation_level_data,
            **protein_level_data,
        }

    @staticmethod
    def preprocess(data_path, prefix, pocket_mode, device):
        print('Preprocessing optimised version of the dataset')
        protein_level_data = {}
        fragmentation_level_data = []

        table_path = os.path.join(data_path, f'{prefix}_table.csv')
        fragments_path = os.path.join(data_path, f'{prefix}_frag.sdf')
        linkers_path = os.path.join(data_path, f'{prefix}_link.sdf')
        pockets_path = os.path.join(data_path, f'{prefix}_pockets.pkl')

        is_geom = True
        is_multifrag = 'multifrag' in prefix

        with open(pockets_path, 'rb') as f:
            pockets = pickle.load(f)

        table = pd.read_csv(table_path)
        generator = tqdm(
            zip(table.iterrows(), read_sdf(fragments_path), read_sdf(linkers_path), pockets),
            total=len(table)
        )
        for (_, row), fragments, linker, pocket_data in generator:
            uuid = row['uuid']
            name = row['molecule']
            frag_pos, frag_one_hot, frag_charges = parse_molecule(fragments, is_geom=is_geom)
            link_pos, link_one_hot, link_charges = parse_molecule(linker, is_geom=is_geom)

            # Parsing pocket data
            pocket_pos = pocket_data[f'{pocket_mode}_coord']
            pocket_one_hot = []
            pocket_charges = []
            for atom_type in pocket_data[f'{pocket_mode}_types']:
                pocket_one_hot.append(get_one_hot(atom_type, const.GEOM_ATOM2IDX))
                pocket_charges.append(const.GEOM_CHARGES[atom_type])
            pocket_one_hot = np.array(pocket_one_hot)
            pocket_charges = np.array(pocket_charges)

            positions = np.concatenate([frag_pos, pocket_pos, link_pos], axis=0)
            one_hot = np.concatenate([frag_one_hot, pocket_one_hot, link_one_hot], axis=0)
            charges = np.concatenate([frag_charges, pocket_charges, link_charges], axis=0)
            anchors = np.zeros_like(charges)

            if is_multifrag:
                for anchor_idx in map(int, row['anchors'].split('-')):
                    anchors[anchor_idx] = 1
            else:
                anchors[row['anchor_1']] = 1
                anchors[row['anchor_2']] = 1

            fragment_only_mask = np.concatenate([
                np.ones_like(frag_charges),
                np.zeros_like(pocket_charges),
                np.zeros_like(link_charges)
            ])
            pocket_mask = np.concatenate([
                np.zeros_like(frag_charges),
                np.ones_like(pocket_charges),
                np.zeros_like(link_charges)
            ])
            linker_mask = np.concatenate([
                np.zeros_like(frag_charges),
                np.zeros_like(pocket_charges),
                np.ones_like(link_charges)
            ])
            fragment_mask = np.concatenate([
                np.ones_like(frag_charges),
                np.ones_like(pocket_charges),
                np.zeros_like(link_charges)
            ])

            fragmentation_level_data.append({
                'uuid': uuid,
                'name': name,
                'anchors': torch.tensor(anchors, dtype=const.TORCH_FLOAT, device=device),
                'fragment_only_mask': torch.tensor(fragment_only_mask, dtype=const.TORCH_FLOAT, device=device),
                'pocket_mask': torch.tensor(pocket_mask, dtype=const.TORCH_FLOAT, device=device),
                'fragment_mask': torch.tensor(fragment_mask, dtype=const.TORCH_FLOAT, device=device),
                'linker_mask': torch.tensor(linker_mask, dtype=const.TORCH_FLOAT, device=device),
            })
            protein_level_data[name] = {
                'positions': torch.tensor(positions, dtype=const.TORCH_FLOAT, device=device),
                'one_hot': torch.tensor(one_hot, dtype=const.TORCH_FLOAT, device=device),
                'charges': torch.tensor(charges, dtype=const.TORCH_FLOAT, device=device),
                'num_atoms': len(positions),
            }

        return {
            'fragmentation_level_data': fragmentation_level_data,
            'protein_level_data': protein_level_data,
        }


def collate(batch):
    out = {}

    # Filter out big molecules
    # if 'pocket_mask' not in batch[0].keys():
    #    batch = [data for data in batch if data['num_atoms'] <= 50]
    # else:
    #    batch = [data for data in batch if data['num_atoms'] <= 1000]

    for i, data in enumerate(batch):
        for key, value in data.items():
            out.setdefault(key, []).append(value)

    for key, value in out.items():
        if key in const.DATA_LIST_ATTRS:
            continue
        if key in const.DATA_ATTRS_TO_PAD:
            out[key] = torch.nn.utils.rnn.pad_sequence(value, batch_first=True, padding_value=0)
            continue
        raise Exception(f'Unknown batch key: {key}')

    atom_mask = (out['fragment_mask'].bool() | out['linker_mask'].bool()).to(const.TORCH_INT)
    out['atom_mask'] = atom_mask[:, :, None]

    batch_size, n_nodes = atom_mask.size()

    # In case of MOAD edge_mask is batch_idx
    if 'pocket_mask' in batch[0].keys():
        batch_mask = torch.cat([
            torch.ones(n_nodes, dtype=const.TORCH_INT) * i
            for i in range(batch_size)
        ]).to(atom_mask.device)
        out['edge_mask'] = batch_mask
    else:
        edge_mask = atom_mask[:, None, :] * atom_mask[:, :, None]
        diag_mask = ~torch.eye(edge_mask.size(1), dtype=const.TORCH_INT, device=atom_mask.device).unsqueeze(0)
        edge_mask *= diag_mask
        out['edge_mask'] = edge_mask.view(batch_size * n_nodes * n_nodes, 1)

    for key in const.DATA_ATTRS_TO_ADD_LAST_DIM:
        if key in out.keys():
            out[key] = out[key][:, :, None]

    return out


def collate_with_fragment_edges(batch):
    out = {}

    # Filter out big molecules
    # batch = [data for data in batch if data['num_atoms'] <= 50]

    for i, data in enumerate(batch):
        for key, value in data.items():
            out.setdefault(key, []).append(value)

    for key, value in out.items():
        if key in const.DATA_LIST_ATTRS:
            continue
        if key in const.DATA_ATTRS_TO_PAD:
            out[key] = torch.nn.utils.rnn.pad_sequence(value, batch_first=True, padding_value=0)
            continue
        raise Exception(f'Unknown batch key: {key}')

    frag_mask = out['fragment_mask']
    edge_mask = frag_mask[:, None, :] * frag_mask[:, :, None]
    diag_mask = ~torch.eye(edge_mask.size(1), dtype=const.TORCH_INT, device=frag_mask.device).unsqueeze(0)
    edge_mask *= diag_mask

    batch_size, n_nodes = frag_mask.size()
    out['edge_mask'] = edge_mask.view(batch_size * n_nodes * n_nodes, 1)

    # Building edges and covalent bond values
    rows, cols, bonds = [], [], []
    for batch_idx in range(batch_size):
        for i in range(n_nodes):
            for j in range(n_nodes):
                rows.append(i + batch_idx * n_nodes)
                cols.append(j + batch_idx * n_nodes)

    edges = [torch.LongTensor(rows).to(frag_mask.device), torch.LongTensor(cols).to(frag_mask.device)]
    out['edges'] = edges

    atom_mask = (out['fragment_mask'].bool() | out['linker_mask'].bool()).to(const.TORCH_INT)
    out['atom_mask'] = atom_mask[:, :, None]

    for key in const.DATA_ATTRS_TO_ADD_LAST_DIM:
        if key in out.keys():
            out[key] = out[key][:, :, None]

    return out


def collate_with_fragment_without_pocket_edges(batch):
    out = {}

    # Filter out big molecules
    # batch = [data for data in batch if data['num_atoms'] <= 50]

    for i, data in enumerate(batch):
        for key, value in data.items():
            out.setdefault(key, []).append(value)

    for key, value in out.items():
        if key in const.DATA_LIST_ATTRS:
            continue
        if key in const.DATA_ATTRS_TO_PAD:
            out[key] = torch.nn.utils.rnn.pad_sequence(value, batch_first=True, padding_value=0)
            continue
        raise Exception(f'Unknown batch key: {key}')

    frag_mask = out['fragment_only_mask']
    edge_mask = frag_mask[:, None, :] * frag_mask[:, :, None]
    diag_mask = ~torch.eye(edge_mask.size(1), dtype=const.TORCH_INT, device=frag_mask.device).unsqueeze(0)
    edge_mask *= diag_mask

    batch_size, n_nodes = frag_mask.size()
    out['edge_mask'] = edge_mask.view(batch_size * n_nodes * n_nodes, 1)

    # Building edges and covalent bond values
    rows, cols, bonds = [], [], []
    for batch_idx in range(batch_size):
        for i in range(n_nodes):
            for j in range(n_nodes):
                rows.append(i + batch_idx * n_nodes)
                cols.append(j + batch_idx * n_nodes)

    edges = [torch.LongTensor(rows).to(frag_mask.device), torch.LongTensor(cols).to(frag_mask.device)]
    out['edges'] = edges

    atom_mask = (out['fragment_mask'].bool() | out['linker_mask'].bool()).to(const.TORCH_INT)
    out['atom_mask'] = atom_mask[:, :, None]

    for key in const.DATA_ATTRS_TO_ADD_LAST_DIM:
        if key in out.keys():
            out[key] = out[key][:, :, None]

    return out


def get_dataloader(dataset, batch_size, collate_fn=collate, shuffle=False):
    return DataLoader(dataset, batch_size, collate_fn=collate_fn, shuffle=shuffle)


def create_template(tensor, fragment_size, linker_size, fill=0):
    values_to_keep = tensor[:fragment_size]
    values_to_add = torch.ones(linker_size, tensor.shape[1], dtype=values_to_keep.dtype, device=values_to_keep.device)
    values_to_add = values_to_add * fill
    return torch.cat([values_to_keep, values_to_add], dim=0)


def create_templates_for_linker_generation(data, linker_sizes):
    """
    Takes data batch and new linker size and returns data batch where fragment-related data is the same
    but linker-related data is replaced with zero templates with new linker sizes
    """
    decoupled_data = []
    for i, linker_size in enumerate(linker_sizes):
        data_dict = {}
        fragment_mask = data['fragment_mask'][i].squeeze()
        fragment_size = fragment_mask.sum().int()
        for k, v in data.items():
            if k == 'num_atoms':
                # Computing new number of atoms (fragment_size + linker_size)
                data_dict[k] = fragment_size + linker_size
                continue
            if k in const.DATA_LIST_ATTRS:
                # These attributes are written without modification
                data_dict[k] = v[i]
                continue
            if k in const.DATA_ATTRS_TO_PAD:
                # Should write fragment-related data + (zeros x linker_size)
                fill_value = 1 if k == 'linker_mask' else 0
                template = create_template(v[i], fragment_size, linker_size, fill=fill_value)
                if k in const.DATA_ATTRS_TO_ADD_LAST_DIM:
                    template = template.squeeze(-1)
                data_dict[k] = template

        decoupled_data.append(data_dict)

    return collate(decoupled_data)

if __name__ == "__main__":
    # val_data_prefix =  'geom_multifrag_test'
    # data_path = r'C:\Users\ziv-admin\PycharmProjects\semla-flow\geom_data'
    # device = 'cpu'
    # test_dataset = ZincDataset(data_path, val_data_prefix,device)
    # batch_size = 1
    # test_dataloader = get_dataloader(test_dataset, batch_size, collate_fn=collate)
    #
    # true_molecules_sdf_path = "./../../geom_data/true_molecules.sdf"
    # true_fragments_sdf_path = "./../../geom_data/true_fragments.sdf"
    #
    # # Initialize SD writers
    # true_molecules_writer = Chem.SDWriter(true_molecules_sdf_path)
    # true_fragments_writer = Chem.SDWriter(true_fragments_sdf_path)
    #
    # for b, data in tqdm(enumerate(test_dataloader), total=len(test_dataloader), desc='Sampling'):
    #     atom_mask = data['atom_mask']
    #     fragment_mask = data['fragment_mask']
    #     true_molecules_batch = build_molecules(
    #         data['one_hot'],
    #         data['positions'],
    #         atom_mask,
    #         is_geom=True,
    #     )
    #     true_fragments_batch = build_molecules(
    #         data['one_hot'],
    #         data['positions'],
    #         fragment_mask,
    #         is_geom=True,
    #     )
    #
    #     # Write true molecules to SDF
    #     for mol in true_molecules_batch:
    #         if mol is not None:  # Ensure the molecule is valid
    #             true_molecules_writer.write(mol)
    #
    #     # Write true fragments to SDF
    #     for frag in true_fragments_batch:
    #         if frag is not None:  # Ensure the fragment is valid
    #             true_fragments_writer.write(frag)
    #
    # # Close the writers
    # true_molecules_writer.close()
    # true_fragments_writer.close()
    #
    # print(f"Saved true molecules to {true_molecules_sdf_path}")
    # print(f"Saved true fragments to {true_fragments_sdf_path}")
    from semlaflow.preprocess import *

    supplier = Chem.SDMolSupplier(r'C:\Users\ziv-admin\PycharmProjects\semla-flow\geom_data\difflinker_data\true_molecules_with_h.sdf',
                                  sanitize=False)
    true_molecules_batch = [mol for mol in supplier if mol is not None]

    supplier = Chem.SDMolSupplier(r'C:\Users\ziv-admin\PycharmProjects\semla-flow\geom_data\difflinker_data\true_fragments_with_h.sdf',
                                  sanitize=False)
    true_fragments_batch = [mol for mol in supplier if mol is not None]
    fragments = []
    for real_mol, frag_mol in zip(true_molecules_batch, true_fragments_batch):
        real_size = real_mol.GetNumAtoms()
        frag_geom = GeometricMol.from_rdkit(frag_mol)
        max_atoms = max(real_size, frag_geom.coords.shape[0])
        frag_geom = GeometricMol.pad_molecule(frag_geom, max_atoms)
        fragments.append(frag_geom)

    # smol_mols = [GeometricMol.from_rdkit(raw_mol) for raw_mol in molecules]
    batch = GeometricMolBatch.from_list(fragments)
    dataset_bytes = batch.to_bytes()
    Path(r'C:\Users\ziv-admin\PycharmProjects\semla-flow\geom_data\fragments.smol').write_bytes(dataset_bytes)