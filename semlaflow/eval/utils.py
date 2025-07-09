from rdkit import Chem
import json
def create_dummy_molecule(coords):
    # Create an RDKit molecule object
    mol = Chem.RWMol()
    # Add dummy atoms to the molecule using the provided coordinates
    for coord in coords:
        x, y, z = coord
        atom = Chem.Atom(0)  # 0 represents a dummy atom
        atom.SetNoImplicit(True)  # Set to avoid implicit valence errors
        atom_idx = mol.AddAtom(atom)  # Add the atom to the molecule
    conf = Chem.Conformer(len(coords))
    for i, coord in enumerate(coords):
        x, y, z = coord
        conf.SetAtomPosition(i, (float(x), float(y), float(z)))  # Set atom position
    mol.AddConformer(conf, assignId=True)  # Add conformer to the molecule
    return mol

from rdkit import Chem
from rdkit.Chem import AllChem

def create_dummy_molecule_with_properties(coords, match_fractions):
    """
    coords:    list of (x,y,z)
    match_fractions: list of floats, one per coord
    """
    mol = Chem.RWMol()
    # add one dummy atom per coord
    for _ in coords:
        atom = Chem.Atom(0)  # atomic number 0 = dummy
        atom.SetNoImplicit(True)
        mol.AddAtom(atom)

    # add 3D conformer
    conf = Chem.Conformer(len(coords))
    for i, (x, y, z) in enumerate(coords):
        conf.SetAtomPosition(i, (float(x), float(y), float(z)))
    mol.AddConformer(conf, assignId=True)

    # encode all match_fractions as a JSON list on the molecule
    mol.SetProp(
        "match_fractions",
        json.dumps([float(mf) for mf in match_fractions])
    )
    return mol


def create_dummy_molecule_with_bfactor(coords, match_fractions):
    """
    Create a dummy RDKit molecule with specified 3D coordinates and B-factors.

    Parameters:
    - coords: List of (x, y, z) tuples for atom positions.
    - match_fractions: List of B-factor values (floats) for each atom.

    Returns:
    - mol: RDKit Mol object with conformer and B-factors set.
    """
    # 1) Build an RWMol
    rw = Chem.RWMol()
    for _ in coords:
        a = Chem.Atom(0)  # Atomic number 0 = dummy atom
        a.SetNoImplicit(True)
        rw.AddAtom(a)
    mol = rw.GetMol()

    # 2) Sanitize before adding PDB info
    Chem.SanitizeMol(mol)

    # 3) Add a conformer with coordinates
    conf = Chem.Conformer(len(coords))
    for i, (x, y, z) in enumerate(coords):
        conf.SetAtomPosition(i, (float(x), float(y), float(z)))
    mol.AddConformer(conf, assignId=True)

    # 4) Set PDBResidueInfo and B-factors
    for i, mf in enumerate(match_fractions):
        atom = mol.GetAtomWithIdx(i)
        info = atom.GetPDBResidueInfo()
        if info is None:
            info = Chem.AtomPDBResidueInfo()
            info.SetResidueName("DUM")
            info.SetName(f"A{i:02d}")
        info.SetTempFactor(float(mf))
        atom.SetMonomerInfo(info)  # Reassign updated info

    return mol



def load_mols_from_sdf(path):
    """Load RDKit Mol objects (with 3D coords) from an SDF, filtering out None."""
    sup = Chem.SDMolSupplier(path, removeHs=False)
    return [m for m in sup if m is not None]