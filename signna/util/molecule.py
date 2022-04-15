import numpy
import torch
from torch_geometric.data import Data


# Categories of hybridization of the atoms.
cat_hbd = ['SP', 'SP2', 'SP3', 'SP3D', 'SP3D2']

# Categories of formal charge of the atoms.
cat_fc = ['-4', '-3', '-2', '-1', '0', '1', '2', '3', '4']

# Bond types.
cat_bond_types = [
    'UNSPECIFIED',
    'SINGLE',
    'DOUBLE',
    'TRIPLE',
    'QUADRUPLE',
    'QUINTUPLE',
    'HEXTUPLE',
    'ONEANDAHALF',
    'TWOANDAHALF',
    'THREEANDAHALF',
    'FOURANDAHALF',
    'FIVEANDAHALF',
    'AROMATIC',
    'IONIC',
    'HYDROGEN',
    'THREECENTER',
    'DATIVEONE',
    'DATIVE',
    'DATIVEL',
    'DATIVER',
    'OTHER',
    'ZERO',
    'SELF'
]


def get_one_hot_feat(hot_category, categories):

    one_hot_feat = dict()
    for cat in categories:
        one_hot_feat[cat] = 0

    # Set value for the selected category if it exists in the given available categories.
    if hot_category in categories:
        one_hot_feat[hot_category] = 1

    # Convert the one-hot encoding dictionary to the one-hot encoding vector.
    return numpy.array(list(one_hot_feat.values()))


def get_mol_graph(mol, elem_feats, target):
    try:
        if mol is None:
            return None

        # Global information of the molecule.
        n_rings = mol.GetRingInfo().NumRings()

        # Structural information of the molecular graph.
        atom_feats = list()
        bonds = list()
        bond_feats = list()

        # Generate node-feature matrix.
        for atom in mol.GetAtoms():
            # Get elemental features of the atom.
            elem_attr = elem_feats[atom.GetAtomicNum() - 1, :]

            # Get hybridization type of the atom.
            hbd_type = get_one_hot_feat(str(atom.GetHybridization()), cat_hbd)

            # Get formal charge of the atom.
            fc_type = get_one_hot_feat(str(atom.GetFormalCharge()), cat_fc)

            # Check whether the atom belongs to the aromatic ring in the molecule.
            mem_aromatic = 1 if atom.GetIsAromatic() else 0

            # Get the number of bonds.
            degree = atom.GetDegree()

            # Get the number of hydrogen bonds.
            n_hs = atom.GetTotalNumHs()

            # Append a feature vector of the atom.
            atom_feats.append(numpy.hstack([elem_attr, hbd_type, fc_type, mem_aromatic, degree, n_hs, n_rings]))

        # Generate bond-feature matrix.
        for bond in mol.GetBonds():
            bonds.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
            bond_feats.append(get_one_hot_feat(str(bond.GetBondType()), cat_bond_types))
            bonds.append([bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()])
            bond_feats.append(get_one_hot_feat(str(bond.GetBondType()), cat_bond_types))

        # Add self-loop.
        for i in range(0, len(atom_feats)):
            bonds.append([i, i])
            bond_feats.append(get_one_hot_feat('SELF', cat_bond_types))

        # Check isolated graph and raise a run time error.
        if len(bonds) == 0:
            return None

        # Convert numpy.ndarray to torch.Tensor for graph-based machine learning.
        atom_feats = torch.tensor(atom_feats, dtype=torch.float)
        bonds = torch.tensor(bonds, dtype=torch.long).t().contiguous()
        bond_feats = torch.tensor(bond_feats, dtype=torch.float)
        target = torch.tensor(target, dtype=torch.float).view(1, 1)

        return Data(x=atom_feats, edge_index=bonds, edge_attr=bond_feats, target=target)
    except AssertionError:
        return None
