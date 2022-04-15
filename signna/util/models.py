import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from pandas import DataFrame
from torch_geometric.loader import DataLoader
from torch_geometric.nn import TransformerConv
from torch_geometric.nn import CGConv
from torch_geometric.nn import GCNConv
from torch_geometric.nn import NNConv
from torch_geometric.nn import LayerNorm
from torch_geometric.nn import global_mean_pool
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score


def exec_signna(exp_idx, dataset_train, dataset_test, gnns, path_results, batch_size, init_lr, n_epochs):
    exp_id = 'signna_' + str(exp_idx)
    train_losses = list()
    dim_emb = 32
    l2_coeff = 5e-6

    targets_test = numpy.vstack([d[0].y.item() for d in dataset_test])
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    loader_test = DataLoader(dataset_test, batch_size=batch_size)

    gnn_models = list()
    for i in range(0, len(gnns)):
        if gnns[i] == 'tfgnn':
            gnn_models.append(TFGNN(n_node_feats=dataset_train[0][i].x.shape[1],
                                    n_edge_feats=dataset_train[0][i].edge_attr.shape[1],
                                    dim_out=dim_emb))
        elif gnns[i] == 'cgcnn':
            gnn_models.append(CGCNN(n_node_feats=dataset_train[0][i].x.shape[1],
                                    n_edge_feats=dataset_train[0][i].edge_attr.shape[1],
                                    dim_out=dim_emb))

    model = SIGNNA(gnns=gnn_models, dim_emb=dim_emb, dim_out=1).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr, weight_decay=l2_coeff)
    criterion = torch.nn.L1Loss()

    print('----------- ' + exp_id + ' -----------')
    for epoch in range(0, n_epochs):
        train_loss = model.fit(loader_train, optimizer, criterion)
        preds_test = model.test(loader_test)
        test_mae = mean_absolute_error(targets_test, preds_test)
        test_r2 = r2_score(targets_test, preds_test)
        print('Epoch [{}/{}]\tTrain loss: {:.4f}\tTest R2: {:.4f}'.format(epoch + 1, n_epochs, train_loss, test_r2))

        train_losses.append([epoch + 1, train_loss, test_mae])

    preds_test = model.test(loader_test)
    test_mae = mean_absolute_error(targets_test, preds_test)
    test_r2 = r2_score(targets_test, preds_test)
    print(test_mae, test_r2)

    DataFrame(train_losses).to_excel(path_results + '/train_losses_' + exp_id + '.xlsx', index=None, header=None)

    pred_results = numpy.hstack([targets_test, preds_test])
    DataFrame(pred_results).to_excel(path_results + '/pred_results_' + exp_id + '.xlsx', index=None, header=None)

    return test_mae, test_r2


class GCN(nn.Module):
    def __init__(self, n_node_feats, dim_out):
        super(GCN, self).__init__()
        self.fc1 = nn.Linear(n_node_feats, 128)
        self.gc1 = GCNConv(in_channels=128, out_channels=128)
        self.gn1 = LayerNorm(128)
        self.gc2 = GCNConv(in_channels=128, out_channels=128)
        self.gn2 = LayerNorm(128)
        self.fc2 = nn.Linear(128, dim_out)

    def forward(self, g):
        hx = self.fc1(g.x)
        h = F.relu(self.gn1(self.gc1(hx, g.edge_index)))
        h = F.relu(self.gn2(self.gc2(h, g.edge_index)))
        out = self.fc2(h)

        return out

    def fit(self, data_loader, optimizer, criterion):
        self.train()
        train_loss = 0

        for batch in data_loader:
            batch = batch.cuda()

            preds = self(batch)
            loss = criterion(batch.y, preds)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.detach().item()

        return train_loss / len(data_loader)

    def test(self, data_loader):
        self.eval()
        list_preds = list()

        with torch.no_grad():
            for batch in data_loader:
                batch = batch.cuda()

                preds = self(batch)
                list_preds.append(preds)

        return torch.vstack(list_preds).cpu().numpy()


class ECCN(nn.Module):
    def __init__(self, n_node_feats, n_edge_feats, dim_out):
        super(ECCN, self).__init__()
        self.fc1 = nn.Linear(n_node_feats, 64)
        self.fce1 = nn.Sequential(nn.Linear(n_edge_feats, 64 * 64), nn.ReLU())
        self.gc1 = NNConv(in_channels=64, out_channels=64, nn=self.fce1)
        self.gn1 = LayerNorm(64)
        self.fce2 = nn.Sequential(nn.Linear(n_edge_feats, 64 * 64), nn.ReLU())
        self.gc2 = NNConv(in_channels=64, out_channels=64, nn=self.fce2)
        self.gn2 = LayerNorm(64)
        self.fc2 = nn.Linear(64, dim_out)

    def forward(self, g):
        hx = self.fc1(g.x)
        h = F.relu(self.gn1(self.gc1(hx, g.edge_index, g.edge_attr)))
        h = F.relu(self.gn2(self.gc2(h, g.edge_index, g.edge_attr)))
        out = self.fc2(h)

        return out

    def fit(self, data_loader, optimizer, criterion):
        self.train()
        train_loss = 0

        for batch in data_loader:
            batch = batch.cuda()

            preds = self(batch)
            loss = criterion(batch.y, preds)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.detach().item()

        return train_loss / len(data_loader)

    def test(self, data_loader):
        self.eval()
        list_preds = list()

        with torch.no_grad():
            for batch in data_loader:
                batch = batch.cuda()

                preds = self(batch)
                list_preds.append(preds)

        return torch.vstack(list_preds).cpu().numpy()


class TFGNN(nn.Module):
    def __init__(self, n_node_feats, n_edge_feats, dim_out):
        super(TFGNN, self).__init__()
        self.fc1 = nn.Linear(n_node_feats, 128)
        self.gc1 = TransformerConv(in_channels=128, out_channels=128, edge_dim=n_edge_feats)
        self.gn1 = LayerNorm(128)
        self.gc2 = TransformerConv(in_channels=128, out_channels=128, edge_dim=n_edge_feats)
        self.gn2 = LayerNorm(128)
        self.fc2 = nn.Linear(128, dim_out)

    def forward(self, g):
        hx = self.fc1(g.x)
        h = F.relu(self.gn1(self.gc1(hx, g.edge_index, g.edge_attr)))
        h = F.relu(self.gn2(self.gc2(h, g.edge_index, g.edge_attr)))
        out = self.fc2(h)

        return out

    def fit(self, data_loader, optimizer, criterion):
        self.train()
        train_loss = 0

        for batch in data_loader:
            batch = batch.cuda()

            preds = self(batch)
            loss = criterion(batch.y, preds)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.detach().item()

        return train_loss / len(data_loader)

    def test(self, data_loader):
        self.eval()
        list_preds = list()

        with torch.no_grad():
            for batch in data_loader:
                batch = batch.cuda()

                preds = self(batch)
                list_preds.append(preds)

        return torch.vstack(list_preds).cpu().numpy()


class CGCNN(nn.Module):
    def __init__(self, n_node_feats, n_edge_feats, dim_out):
        super(CGCNN, self).__init__()
        self.fc1 = nn.Linear(n_node_feats, 128)
        self.gc1 = CGConv(128, n_edge_feats)
        self.gn1 = LayerNorm(128)
        self.gc2 = CGConv(128, n_edge_feats)
        self.gn2 = LayerNorm(128)
        self.fc2 = nn.Linear(128, dim_out)

    def forward(self, g):
        hx = self.fc1(g.x)
        h = F.relu(self.gn1(self.gc1(hx, g.edge_index, g.edge_attr)))
        h = F.relu(self.gn2(self.gc2(h, g.edge_index, g.edge_attr)))
        out = self.fc2(h)

        return out

    def fit(self, data_loader, optimizer, criterion):
        self.train()
        train_loss = 0

        for batch in data_loader:
            batch = batch.cuda()

            preds = self(batch)
            loss = criterion(batch.y, preds)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.detach().item()

        return train_loss / len(data_loader)

    def test(self, data_loader):
        self.eval()
        list_preds = list()

        with torch.no_grad():
            for batch in data_loader:
                batch = batch.cuda()

                preds = self(batch)
                list_preds.append(preds)

        return torch.vstack(list_preds).cpu().numpy()


class SIGNNA(nn.Module):
    def __init__(self, gnns, dim_emb, dim_out):
        super(SIGNNA, self).__init__()
        self.gnns = nn.ModuleList(gnns)
        self.fc1 = nn.Linear(len(gnns) * dim_emb, 64)
        self.fc2 = nn.Linear(64, dim_out)

    def forward(self, g_list):
        h = [global_mean_pool(self.gnns[i](g_list[i]), g_list[i].batch) for i in range(0, len(g_list))]
        h = F.relu(self.fc1(torch.hstack(h)))
        out = self.fc2(h)

        return out, h

    def fit(self, data_loader, optimizer, criterion):
        self.train()
        train_loss = 0

        for batch in data_loader:
            for i in range(0, len(batch)):
                batch[i].cuda()

            preds, embs = self(batch)
            loss = criterion(batch[0].y, preds)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.detach().item()

        return train_loss / len(data_loader)

    def test(self, data_loader):
        self.eval()
        list_preds = list()

        with torch.no_grad():
            for batch in data_loader:
                for i in range(0, len(batch)):
                    batch[i].cuda()

                preds, _ = self(batch)
                list_preds.append(preds)

        return torch.vstack(list_preds).cpu().numpy()
