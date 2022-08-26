import numpy
import pandas
import torch
import json
import os
import re
from tqdm import tqdm
from rdkit.Chem import MolFromMolFile
from torch_geometric.data import Batch
from util.chem import load_elem_attrs
from util.crystal import get_mol_graph
from util.crystal import get_crystal_graph


class ChemSystem:
    def __init__(self, struct_react, struct_env, sys_id, y=None):
        self.struct_react = struct_react if isinstance(struct_react, list) else [struct_react]
        self.struct_env = struct_env if isinstance(struct_env, list) else [struct_env]
        self.y = y
        self.sys_id = sys_id


def load_dataset(path_metadata_file, path_structs, idx_target, n_bond_feats, atomic_cutoff=4.0, vn_method=None):
    metadata = numpy.array(pandas.read_excel(path_metadata_file))
    elem_attrs_org = load_elem_attrs('res/matscholar-embedding.json')
    elem_attrs_inorg = load_elem_attrs('res/cgcnn-embedding.json')
    rbf_means = numpy.linspace(start=1.0, stop=atomic_cutoff, num=n_bond_feats)
    dataset = list()

    for i in tqdm(range(0, metadata.shape[0])):
        struct_id = metadata[i, 0]
        g_org = get_mol_graph(MolFromMolFile(path_structs + '/' + struct_id + '.mol'), elem_attrs_org)
        g_inorg = get_crystal_graph(path_structs + '/' + struct_id + '.cif',
                                    elem_attrs_inorg,
                                    rbf_means,
                                    atomic_cutoff,
                                    vn_method)

        if g_org is not None and g_inorg is not None:
            dataset.append(ChemSystem(g_org, g_inorg, sys_id=i, y=metadata[i, idx_target]))

    return dataset


def load_dataset_cathub(path_metadata_file, path_structs, idx_target, n_bond_feats, atomic_cutoff=4.0):
    metadata = numpy.array(pandas.read_excel(path_metadata_file))
    elem_attrs_react_mol = load_elem_attrs('res/matscholar-embedding.json')
    elem_attrs_react_surf = load_elem_attrs('res/cgcnn-embedding.json')
    elem_attrs_prod = load_elem_attrs('res/cgcnn-embedding.json')
    rbf_means = numpy.linspace(start=1.0, stop=atomic_cutoff, num=n_bond_feats)
    dataset = list()

    for i in tqdm(range(0, metadata.shape[0])):
        sys_id = metadata[i, 0]
        react_ids = list(json.loads(metadata[i, 4].replace('\'', '"')).keys())
        prod_ids = list(json.loads(metadata[i, 5].replace('\'', '"')).keys())

        fname_react_mol = path_structs + '/' + sys_id + '/' + react_ids[0] + '.mol'
        fname_react_surf = path_structs + '/' + sys_id + '/' + react_ids[1] + '.cif'
        fname_prod = path_structs + '/' + sys_id + '/' + prod_ids[0] + '.cif'

        if not validate_files([fname_react_mol, fname_react_surf, fname_prod]):
            continue

        g_react_mol = get_mol_graph(MolFromMolFile(fname_react_mol), elem_attrs_react_mol)
        g_react_surf = get_crystal_graph(fname_react_surf, elem_attrs_react_surf, rbf_means, atomic_cutoff)
        g_prod = get_crystal_graph(fname_prod, elem_attrs_prod, rbf_means, atomic_cutoff)

        if g_react_mol is not None and g_react_surf is not None and g_prod is not None:
            g_reacts = [g_react_mol, g_react_surf]
            dataset.append(ChemSystem(g_reacts, g_prod, i, metadata[i, idx_target]))

    return dataset


def split_dataset(dataset, ratio_train, random_seed=None):
    n_train = int(ratio_train * len(dataset))

    if random_seed is not None:
        numpy.random.seed(random_seed)

    idx_rand = numpy.random.permutation(len(dataset))

    dataset_train = [dataset[idx] for idx in idx_rand[:n_train]]
    dataset_test = [dataset[idx] for idx in idx_rand[n_train:]]

    return dataset_train, dataset_test


def collate(batch):
    n_react_structs = len(batch[0].struct_react)
    n_env_structs = len(batch[0].struct_env)
    structs_react = [list() for i in range(0, n_react_structs)]
    structs_env = [list() for i in range(0, n_env_structs)]
    y = list()

    for b in batch:
        structs_react.append(b.struct_react)
        structs_env.append(b.struct_env)
        y.append(torch.tensor(b.y, dtype=torch.float))

        for i in range(0, n_react_structs):
            structs_react[i].append(b.struct_react[i])

        for i in range(0, n_env_structs):
            structs_env[i].append(b.struct_env[i])

    structs_react = [Batch.from_data_list(structs_react[i]) for i in range(0, n_react_structs)]
    structs_env = [Batch.from_data_list(structs_env[i]) for i in range(0, n_env_structs)]
    y = torch.vstack(y)

    return structs_react, structs_env, y


def validate_files(file_names):
    for fname in file_names:
        if not os.path.isfile(fname):
            return False

    return True


def sample_cathub_dataset(dataset_name, n_reacts, n_prods, n_elems):
    dataset = list()

    for i in range(0, 6):
        print(i)
        data = numpy.array(pandas.read_excel('../../data/chem_data/cathub/metadata_' + str(i) + '.xlsx'))

        for j in range(0, data.shape[0]):
            reacts = json.loads(data[j, 4].replace('\'', '\"'))
            prods = json.loads(data[j, 5].replace('\'', '\"'))
            form = data[j, 3].split(' -> ')[0].split(' + ')[0]

            if len(reacts) == n_reacts and len(prods) == n_prods and len(re.findall(r'[A-Z]', form)) >= n_elems - 1:
                dataset.append(data[j])

    pandas.DataFrame(dataset).to_excel(dataset_name, index=False, header=False)
