import numpy
import json
import torch
import warnings
from rdkit.Chem import MolFromMolFile
from mendeleev.fetch import fetch_table
from pymatgen.core.structure import Structure
from sklearn.metrics import pairwise_distances
from torch_geometric.data import Data
from util.molecule import get_mol_graph


warnings.filterwarnings(action='ignore')

atom_nums = {
    'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
    'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20,
    'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30,
    'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40,
    'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
    'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60,
    'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70,
    'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80,
    'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90,
    'Pa': 91, 'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98, 'Es': 99, 'Fm': 100
}

atom_syms = {v: k for k, v in atom_nums.items()}

elem_attr_names = [
    'atomic_number',
    'period',
    'en_pauling',
    'covalent_radius_bragg',
    'electron_affinity',
    'atomic_volume',
    'atomic_weight',
    'fusion_heat'
]

first_ion_energies = [
    1312, 2372.3, 520.2, 899.5, 800.6, 1086.5, 1402.3, 1313.9, 1681, 2080.7,
    495.8, 737.7, 577.5, 786.5, 1011.8, 999.6, 1251.2, 1520.6, 418.8, 589.8,
    633.1, 658.8, 650.9, 652.9, 717.3, 762.5, 760.4, 737.1, 745.5, 906.4,
    578.8, 762, 947, 941, 1139.9, 1350.8, 403, 549.5, 600, 640.1,
    652.1, 684.3, 702, 710.2, 719.7, 804.4, 731, 867.8, 558.3, 708.6,
    834, 869.3, 1008.4, 1170.4, 375.7, 502.9, 538.1, 534.4, 527, 533.1,
    540, 544.5, 547.1, 593.4, 565.8, 573, 581, 589.3, 596.7, 603.4,
    523.5, 658.5, 761, 770, 760, 840, 880, 870, 890.1, 1007.1,
    589.4, 715.6, 703, 812.1, 899.003, 1037, 380, 509.3, 499, 587,
    568, 597.6, 604.5, 584.7, 578, 581, 601, 608, 619, 627,
    635, 642, 470, 580, 665, 757, 740, 730, 800, 960,
    1020, 1155, 707.2, 832.2, 538.3, 663.9, 736.9, 860.1, 463.1, 563.3
]

org_ref = [atom_nums[e] for e in ['H', 'C', 'N', 'O', 'F', 'S', 'P', 'Cl', 'Br']]


def rbf(data, mu, beta):
    return numpy.exp(-(data - mu)**2 / beta**2)


def load_mendeleev_attrs():
    # Load chemical and physical attributes of the elements from the Python Mendeleev package.
    elem_attrs = numpy.nan_to_num(numpy.array(fetch_table('elements')[elem_attr_names]))

    # Load first ionization energies for each element.
    ion_energies = numpy.array(first_ion_energies[:elem_attrs.shape[0]]).reshape(-1, 1)

    return numpy.hstack([elem_attrs, ion_energies])


def load_elem_attrs(path_elem_attr: str = None):
    if path_elem_attr is None:
        # Load basic elemental features from the Mendeleev package.
        return load_mendeleev_attrs()
    else:
        # Load the elemental features from the user-defined elemental attributes.
        elem_attrs = list()
        with open(path_elem_attr) as json_file:
            elem_attr = json.load(json_file)
            for elem in atom_nums.keys():
                elem_attrs.append(numpy.array(elem_attr[elem]))

        return numpy.vstack(elem_attrs)


def load_struct(path_struct, elem_attr_table, n_bond_feats=None, rbf_means=None, cutoff_radius=None, target=None):
    if n_bond_feats is None:
        g = MolFromMolFile(path_struct)
        g = get_mol_graph(g, elem_attr_table, target)
    else:
        g = Structure.from_file(path_struct)
        list_nbrs = g.get_all_neighbors(r=cutoff_radius)
        atoms, coords = get_graph_info(g, list_nbrs)
        g = get_crystal_graph(atoms, coords, elem_attr_table, cutoff_radius, target, rbf_means, org_ref)

    return g


def get_graph_info(crystal, list_nbrs):
    coord_dict = dict()
    atoms = list()
    coords = list()

    if hasattr(crystal, 'atomic_numbers'):
        atomic_numbers = crystal.atomic_numbers
    else:
        atomic_numbers = [atom_nums[site.species_string.split(':')[0]] for site in crystal.sites]

    for i in range(0, len(atomic_numbers)):
        coord_key = ','.join(list(crystal.cart_coords[i, :].astype(str)))
        coord_dict[coord_key] = True
        atoms.append(atomic_numbers[i])
        coords.append(crystal.cart_coords[i, :])

    for i in range(0, len(list_nbrs)):
        nbrs = list_nbrs[i]

        for j in range(0, len(nbrs)):
            coord_key = ','.join(list(nbrs[j][0].coords.astype(str)))

            if coord_key not in coord_dict.keys():
                coord_dict[coord_key] = True
                atom_num = atom_nums[nbrs[j][0].species_string.split(':')[0]]
                coord = nbrs[j][0].coords
                atoms.append(atom_num)
                coords.append(coord)

    return atoms, numpy.vstack(coords)


def get_crystal_graph(atoms, coords, elem_attrs, cutoff_radius, target, rbf_means, atoms_org_ref):
    if len(atoms) == 0 or len(coords) == 0:
        return None

    pdists = pairwise_distances(coords)
    atom_feats = list()
    bonds = list()
    bond_feats = list()

    for i in range(0, len(atoms)):
        if atoms[i] in atoms_org_ref:
            atom_feats.append(numpy.zeros(elem_attrs.shape[1]))
        else:
            atom_feats.append(elem_attrs[atoms[i] - 1, :])

    for i in range(0, len(atom_feats)):
        ind_nn = numpy.where(pdists[i, :] < cutoff_radius)[0]
        dists_repeat = numpy.full((rbf_means.shape[0], ind_nn.shape[0]), pdists[0, ind_nn]).transpose()
        bonds.append(numpy.column_stack((numpy.full(ind_nn.shape[0], i), ind_nn)))
        bond_feats.append(rbf(dists_repeat, mu=rbf_means, beta=0.5))
    bonds = numpy.vstack(bonds)
    bond_feats = numpy.vstack(bond_feats)

    if len(bonds) == 0:
        return None
    else:
        atom_feats = torch.tensor(atom_feats, dtype=torch.float)
        bonds = torch.tensor(bonds, dtype=torch.long).t().contiguous()
        bond_feats = torch.tensor(bond_feats, dtype=torch.float)
        n_atoms = torch.tensor(atom_feats.shape[0], dtype=torch.long)
        target = torch.tensor(target, dtype=torch.float).view(1, 1)

        return Data(x=atom_feats, edge_index=bonds, edge_attr=bond_feats, n_atoms=n_atoms, y=target)



