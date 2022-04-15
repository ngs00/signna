import pandas
import ast
from tqdm import tqdm
from util.chem import *


def load_dataset(path_metadata_file, path_structs, idx_struct, idx_target, n_bond_feats, cutoff_radius, org_ref):
    metadata = numpy.array(pandas.read_excel(path_metadata_file))
    elem_attrs_inorg = load_elem_attrs('res/cgcnn-embedding.json')
    elem_attrs_org = load_elem_attrs('res/matscholar-embedding.json')
    rbf_means = numpy.linspace(start=1.0, stop=cutoff_radius, num=n_bond_feats)
    dataset = list()

    for i in range(0, metadata.shape[0]):
        target = metadata[i, idx_target]

        try:
            data = load_struct(path_structs, metadata[i, idx_struct], target, elem_attrs_inorg, elem_attrs_org,
                               cutoff_radius, rbf_means, org_ref)

            if None not in data:
                dataset.append(data)
        except:
            continue

    return dataset


def load_hoip2d_dataset(path_metadata_file, path_structs, idx_struct, idx_target, cutoff_radius=5, n_bond_feats=32):
    metadata = numpy.array(pandas.read_excel(path_metadata_file))
    elem_attrs_inorg = load_elem_attrs('res/cgcnn-embedding.json')
    elem_attrs_org = load_elem_attrs('res/matscholar-embedding.json')
    rbf_means = numpy.linspace(start=1.0, stop=cutoff_radius, num=n_bond_feats)
    dataset = list()

    for i in tqdm(range(0, metadata.shape[0])):
        target = metadata[i, idx_target]

        try:
            g_inorg = load_struct(path_structs + '/' + metadata[i, idx_struct] + '.cif', elem_attrs_inorg,
                                  n_bond_feats=n_bond_feats, rbf_means=rbf_means,
                                  cutoff_radius=cutoff_radius, target=target)
            g_mol = load_struct(path_structs + '/' + metadata[i, idx_struct] + '.mol', elem_attrs_org, target=target)

            if g_inorg is not None and g_mol is not None:
                dataset.append([g_inorg, g_mol])
        except:
            continue

    return dataset


def load_cathub_dataset(path_metadata_file, path_structs, idx_cathub_id, idx_react, idx_prod, idx_target,
                        elem_attr_tables, n_bond_feats, cutoff_radius):
    metadata = numpy.array(pandas.read_excel(path_metadata_file))
    cathub_ids = metadata[:, idx_cathub_id]
    rbf_means = numpy.linspace(start=1.0, stop=cutoff_radius, num=n_bond_feats)
    dataset = list()

    for i in tqdm(range(0, metadata.shape[0])):
        reacts = ast.literal_eval(metadata[i, idx_react])
        prods = ast.literal_eval(metadata[i, idx_prod])
        target = metadata[i, idx_target]

        try:
            substructs = list(reacts.keys()) + list(prods.keys())
            subgraphs = list()

            for j in range(0, len(substructs)):
                substruct = load_struct(path_struct=path_structs + '/' + cathub_ids[i] + '/' + substructs[j] + '.cif',
                                        elem_attr_table=elem_attr_tables[j],
                                        n_bond_feats=n_bond_feats,
                                        rbf_means=rbf_means,
                                        cutoff_radius=cutoff_radius,
                                        target=target)
                subgraphs.append(substruct)

            if len(substructs) == len(subgraphs):
                dataset.append(subgraphs)
        except:
            continue

    return dataset


def split_dataset(dataset, ratio, random_seed=None):
    if isinstance(dataset, numpy.ndarray):
        n_data = dataset.shape[0]
    else:
        n_data = len(dataset)

    n_dataset1 = int(ratio * n_data)

    if random_seed is not None:
        numpy.random.seed(random_seed)

    idx_rand = numpy.random.permutation(n_data)

    if isinstance(dataset, numpy.ndarray):
        dataset1 = dataset[idx_rand[:n_dataset1]]
        dataset2 = dataset[idx_rand[n_dataset1:]]
    else:
        dataset1 = [dataset[idx] for idx in idx_rand[:n_dataset1]]
        dataset2 = [dataset[idx] for idx in idx_rand[n_dataset1:]]

    return dataset1, dataset2
