from pymatgen.core import Structure
from sklearn.metrics import pairwise_distances
from util.chem import *


org_ref = [atom_nums[e] for e in ['H', 'C', 'N', 'O', 'F', 'S', 'P', 'Cl', 'Br']]


def get_crystal_graph(path_cif, elem_attrs, rbf_means, atomic_cutoff=4.0, vn_method=None):
    try:
        crystal = Structure.from_file(path_cif)
        atom_coord, atom_feats, ans = get_atom_info(crystal, elem_attrs, atomic_cutoff, vn_method)
        bonds, bond_feats = get_bond_info(atom_coord, rbf_means, atomic_cutoff)

        if bonds is None:
            return None

        atom_feats = torch.tensor(atom_feats, dtype=torch.float)
        bonds = torch.tensor(bonds, dtype=torch.long).t().contiguous()
        bond_feats = torch.tensor(bond_feats, dtype=torch.float)
        ans = torch.tensor(ans, dtype=torch.long)
        n_atoms = torch.tensor(atom_feats.shape[0], dtype=torch.long).view(1, 1)
        crystal_graph = Data(x=atom_feats, edge_index=bonds, edge_attr=bond_feats, atom_nums=ans, n_atoms=n_atoms)

        return crystal_graph
    except AssertionError:
        return None


def get_atom_info(crystal, elem_attrs, atomic_cutoff, vn_method):
    atoms = list(crystal.atomic_numbers)
    atom_coord = list()
    atom_feats = list()
    list_nbrs = crystal.get_all_neighbors(atomic_cutoff)
    charge = crystal.charge
    density = float(crystal.density)
    volume = crystal.volume

    coords = dict()
    for coord in list(crystal.cart_coords):
        coord_key = ','.join(list(coord.astype(str)))
        coords[coord_key] = True

    for i in range(0, len(list_nbrs)):
        nbrs = list_nbrs[i]

        for j in range(0, len(nbrs)):
            coord_key = ','.join(list(nbrs[j][0].coords.astype(str)))
            if coord_key not in coords.keys():
                coords[coord_key] = True
                atoms.append(atom_nums[nbrs[j][0].species_string])

    for coord in coords.keys():
        atom_coord.append(numpy.array([float(x) for x in coord.split(',')]))
    atom_coord = numpy.vstack(atom_coord)

    for i in range(0, len(atoms)):
        if atoms[i] in org_ref:
            if vn_method is None:
                atom_feats.append(numpy.zeros(elem_attrs.shape[1] + 3))
            else:
                atom_feats.append(vn_method(crystal, atoms[i]))
        else:
            elem_attr = elem_attrs[atoms[i]-1, :]
            atom_feats.append(numpy.hstack([elem_attr, charge, density, volume]))
        # elem_attr = elem_attrs[atoms[i]-1, :]
        # atom_feats.append(numpy.hstack([elem_attr, charge, density, volume]))
    atom_feats = numpy.vstack(atom_feats).astype(float)

    return atom_coord, atom_feats, atoms


def get_bond_info(atom_coord, rbf_means, atomic_cutoff):
    bonds = list()
    bond_feats = list()
    pdist = pairwise_distances(atom_coord)

    for i in range(0, atom_coord.shape[0]):
        for j in range(0, atom_coord.shape[0]):
            if i != j and pdist[i, j] < atomic_cutoff:
                bonds.append([i, j])
                bond_feats.append(rbf(numpy.full(rbf_means.shape[0], pdist[i, j]), rbf_means, beta=0.5))

    if len(bonds) == 0:
        return None, None
    else:
        bonds = numpy.vstack(bonds)
        bond_feats = numpy.vstack(bond_feats)

        return bonds, bond_feats


def get_atom_info_baseline(crystal, elem_attrs, atomic_cutoff):
    try:
        atoms = list(crystal.atomic_numbers)
        atom_coord = list()
        atom_feats = list()
        list_nbrs = crystal.get_all_neighbors(atomic_cutoff)
        charge = crystal.charge
        density = float(crystal.density)
        volume = crystal.volume

        coords = dict()
        for coord in list(crystal.cart_coords):
            coord_key = ','.join(list(coord.astype(str)))
            coords[coord_key] = True

        for i in range(0, len(list_nbrs)):
            nbrs = list_nbrs[i]

            for j in range(0, len(nbrs)):
                coord_key = ','.join(list(nbrs[j][0].coords.astype(str)))
                if coord_key not in coords.keys():
                    coords[coord_key] = True
                    atoms.append(atom_nums[nbrs[j][0].species_string])

        for coord in coords.keys():
            atom_coord.append(numpy.array([float(x) for x in coord.split(',')]))
        atom_coord = numpy.vstack(atom_coord)

        for i in range(0, len(atoms)):
            elem_attr = elem_attrs[atoms[i]-1, :]
            atom_feats.append(numpy.hstack([elem_attr, charge, density, volume]))
        atom_feats = numpy.vstack(atom_feats).astype(float)

        return atom_coord, atom_feats, atoms
    except:
        return None, None, None


# def split_crystal_graph(s, g, elem_attrs_org):
#     edges = g.edge_index.t().numpy()
#     nums = g.atom_nums.numpy()
#     org_labels = list()
#
#     for i in range(0, g.x.shape[0]):
#         nn_atoms = list()
#
#         for j in range(0, edges.shape[0]):
#             if edges[j, 0] == i:
#                 nn_atoms.append(nums[edges[j, 1]])
#
#         if nums[i] in org_ref:
#             is_org_atom = True
#             for idx in nn_atoms:
#                 if idx not in org_ref:
#                     is_org_atom = False
#                     break
#         else:
#             is_org_atom = False
#
#         org_labels.append(is_org_atom)
#
#     org_struct = get_org_substruct(s, org_labels, elem_attrs_org)
#     inorg_struct = get_inorg_substruct(g, org_labels)
#
#     if org_struct is None or inorg_struct is None:
#         return None, None
#
#     return org_struct, inorg_struct
#
#
# def get_org_substruct(s, org_labels, elem_attrs):
#     atoms = list(s.atomic_numbers)
#     org_elems = list()
#     coords_org_elems = list()
#
#     for i in range(0, len(atoms)):
#         if org_labels[i]:
#             org_elems.append(atom_syms[atoms[i]])
#             coords_org_elems.append(numpy.array(s.cart_coords[i, :]))
#
#     if len(org_elems) == 0:
#         return None
#
#     temp_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=16))
#     with open(temp_id + '.xyz', 'w') as f:
#         f.write(str(len(org_elems)) + '\n\n')
#
#         for j in range(0, len(org_elems)):
#             coord_x = str(coords_org_elems[j][0])
#             coord_y = str(coords_org_elems[j][1])
#             coord_z = str(coords_org_elems[j][2])
#             coords = coord_x + '\t' + coord_y + '\t' + coord_z
#             f.write(org_elems[j] + '\t' + coords + '\n')
#
#     mol = list(openbabel.pybel.readfile('xyz', temp_id + '.xyz'))[0]
#     mol = mol.write(format='mol')
#
#     with open(temp_id + '.mol', 'w') as f:
#         f.write(mol)
#
#     mol = MolFromMolFile(temp_id + '.mol')
#
#     os.remove(temp_id + '.xyz')
#     os.remove(temp_id + '.mol')
#
#     return get_mol_graph(mol, elem_attrs)
#
#
# def get_inorg_substruct(g, org_labels):
#     inorg_struct = deepcopy(g)
#
#     for i in range(0, g.x.shape[0]):
#         if not org_labels[i]:
#             inorg_struct.x[i, :] = torch.zeros(g.x.shape[1])
#
#     return inorg_struct
#
#
# def get_dummy_struct(n, n_node_feats, n_edge_feats):
#     node_feats = torch.zeros((n, n_node_feats))
#     edges = list()
#
#     for i in range(0, n):
#         for j in range(0, n):
#             edges.append([i, j])
#
#     edges = torch.tensor(edges, dtype=torch.long).t().contiguous()
#     edge_feats = torch.zeros((edges.shape[0], n_edge_feats))
#
#     return Data(x=node_feats, edge_index=edges, edge_attr=edge_feats)
