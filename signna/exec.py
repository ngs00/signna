import numpy
from util.chem import load_elem_attrs
from util.data import load_hoip2d_dataset
from util.data import split_dataset
from util.models import exec_signna


# Experiment settings
n_repeats = 5
batch_size = 32
init_lr = 5e-4
n_epochs = 500


# Load dataset
dataset = load_hoip2d_dataset(path_metadata_file='dataset/hoip2d/metadata.xlsx',
                              path_structs='dataset/hoip2d',
                              idx_struct=0,
                              idx_target=1,
                              cutoff_radius=4,
                              n_bond_feats=32)

# Model configuration
elem_attr_tables = [
    load_elem_attrs('res/cgcnn-embedding.json'),
    load_elem_attrs('res/matscholar-embedding.json'),
]
gnns = ['cgcnn', 'tfgnn']


# Execute training and evaluation process of the configured model
mae_cgcnn = list()
r2_cgcnn = list()
for i in range(0, n_repeats):
    dataset_train, dataset_test = split_dataset(dataset, ratio=0.8, random_seed=i)
    test_mae, test_r2 = exec_signna(exp_idx=i,
                                    dataset_train=dataset_train,
                                    dataset_test=dataset_test,
                                    gnns=gnns,
                                    path_results='result/hoip2d',
                                    batch_size=batch_size,
                                    init_lr=init_lr,
                                    n_epochs=n_epochs)
    mae_cgcnn.append(test_mae)
    r2_cgcnn.append(test_r2)

# Print evaluation metrics
print('------- Evaluation Metrics -------')
print('MAE', numpy.mean(mae_cgcnn), numpy.std(mae_cgcnn))
print('R2-score', numpy.mean(r2_cgcnn), numpy.std(r2_cgcnn))
