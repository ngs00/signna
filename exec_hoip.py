from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from util.data import *
from ml.models import TFGNN
from ml.models import CGCNN
from ml.models import SIGNNA


dataset_name = 'hoip'
dim_emb = 16
n_epochs = 1000
dataset = load_dataset(path_metadata_file='datasets/' + dataset_name + '/metadata.xlsx',
                       path_structs='datasets/' + dataset_name,
                       idx_target=1,
                       n_bond_feats=128)
torch.save(dataset, 'save/' + dataset_name + '/dataset.pt')
dataset = torch.load('save/' + dataset_name + '/dataset.pt')

list_mae = list()
list_r2 = list()
for n in range(0, 5):
    dataset_train, dataset_test = split_dataset(dataset, ratio_train=0.8, random_seed=n)
    loader_train = DataLoader(dataset_train, batch_size=32, shuffle=True, collate_fn=collate)
    loader_test = DataLoader(dataset_test, batch_size=128, collate_fn=collate)
    y_test = numpy.vstack([d.y for d in dataset_test])

    gnn_org = TFGNN(n_node_feats=dataset[0].struct_react[0].x.shape[1],
                    n_edge_feats=dataset[0].struct_react[0].edge_attr.shape[1],
                    dim_out=dim_emb)
    gnn_inorg = CGCNN(n_node_feats=dataset[0].struct_env[0].x.shape[1],
                      n_edge_feats=dataset[0].struct_env[0].edge_attr.shape[1],
                      dim_out=dim_emb)
    model = SIGNNA(gnn_c=gnn_org, gnn_e=gnn_inorg, dim_emb=dim_emb, dim_out=1).cuda()

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=5e-6)
    criterion = torch.nn.L1Loss()

    for epoch in range(0, n_epochs):
        loss_train = model.fit(loader_train, optimizer, criterion)
        print('Epoch [{}/{}]\tTrain loss: {:.4f}\t'.format(epoch + 1, n_epochs, loss_train))

    preds_test = model.predict(loader_test).cpu().numpy()
    mae_test = mean_absolute_error(y_test, preds_test)
    r2_test = r2_score(y_test, preds_test)
    list_mae.append(mae_test)
    list_r2.append(r2_test)
    print(mae_test, r2_test)

    torch.save(model.state_dict(), 'save/' + dataset_name + 'model_' + str(n) + '.pt')
    results = pandas.DataFrame(numpy.hstack([y_test, preds_test]))
    results.to_excel('save/' + dataset_name + '/preds_' + str(n) + '.xlsx', index=False, header=False)


print(numpy.mean(list_mae), numpy.std(list_mae))
print(numpy.mean(list_r2), numpy.std(list_r2))
