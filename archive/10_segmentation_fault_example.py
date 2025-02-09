import rdkit
from rdkit.Chem.rdDistGeom import EmbedMultipleConfs
from rdkit.Chem.rdmolfiles import MolFromMolBlock
from rdkit.Chem.rdmolops import AddHs

MOLBLOCK = """segmentation_fault
     RDKit          3D

 14 16  0  0  1  0  0  0  0  0999 V2000
   -2.6383   -1.3457   -2.3147 C   0  0  2  0  0  0  0  0  0  0  0  0
   -2.6416    0.2493   -2.4783 C   0  0  2  0  0  0  0  0  0  0  0  0
   -1.4682    0.5200    0.1489 N   0  0  0  0  0  0  0  0  0  0  0  0
   -2.1790   -1.5540   -0.8344 C   0  0  0  0  0  0  0  0  0  0  0  0
   -1.7790    3.4105   -1.7076 N   0  0  0  0  0  0  0  0  0  0  0  0
   -1.0867    2.1687   -1.5159 C   0  0  2  0  0  0  0  0  0  0  0  0
   -2.2251   -0.8303   -3.3823 N   0  0  0  0  0  2  0  0  0  0  0  0
   -1.3466    1.1554   -2.5476 N   0  0  0  0  0  0  0  0  0  0  0  0
   -1.6987   -0.4961   -0.2729 N   0  0  0  0  0  0  0  0  0  0  0  0
   -3.2293   -2.6650   -2.6584 O   0  0  0  0  0  0  0  0  0  0  0  0
   -2.3198   -2.6432   -0.3452 O   0  0  0  0  0  0  0  0  0  0  0  0
   -1.3334    1.7234   -0.1738 N   0  0  0  0  0  0  0  0  0  0  0  0
   -1.1653   -0.2280    0.9964 N   0  0  0  0  0  0  0  0  0  0  0  0
   -3.7484    0.9579   -2.1397 O   0  0  0  0  0  1  0  0  0  0  0  0
  1 10  1  6
  1  4  1  0
  2  1  1  0
  2 14  1  6
  3  9  1  0
  3 13  1  1
  4 11  2  0
  4  9  1  0
  6  5  1  6
 12  6  1  0
  7  2  1  0
  7  1  1  0
  8  2  1  0
  8  6  1  0
  9 13  1  0
 12  3  1  0
M  RAD  2   7   2  14   2
M  END
"""

print("RDKit version:", rdkit.__version__)
mol = AddHs(MolFromMolBlock(MOLBLOCK))
EmbedMultipleConfs(mol, numConfs=10)
