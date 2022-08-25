import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv
from torch_geometric.nn import GINConv
from torch_geometric.nn import NNConv
from torch_geometric.nn import TransformerConv
from torch_geometric.nn import CGConv
from torch_geometric.nn import LayerNorm
from torch_geometric.nn import global_mean_pool


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
        h = global_mean_pool(h, g.batch)
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

    def predict(self, data_loader):
        self.eval()

        with torch.no_grad():
            preds = [self(batch.cuda()) for batch in data_loader]

        return torch.vstack(preds)

    def emb(self, data_loader):
        embs = list()

        self.eval()
        with torch.no_grad():
            for b in data_loader:
                b = b.cuda()
                hx = self.fc1(b.x)
                h = F.relu(self.gn1(self.gc1(hx, b.edge_index, b.edge_attr)))
                h = F.relu(self.gn2(self.gc2(h, b.edge_index, b.edge_attr)))
                e = global_mean_pool(h, b.batch)
                embs.append(e)

        return torch.vstack(embs)


class GAT(nn.Module):
    def __init__(self, n_node_feats, dim_out):
        super(GAT, self).__init__()
        self.fc1 = nn.Linear(n_node_feats, 128)
        self.gc1 = GATConv(in_channels=128, out_channels=128)
        self.gn1 = LayerNorm(128)
        self.gc2 = GATConv(in_channels=128, out_channels=128)
        self.gn2 = LayerNorm(128)
        self.fc2 = nn.Linear(128, dim_out)

    def forward(self, g):
        hx = self.fc1(g.x)
        h = F.relu(self.gn1(self.gc1(hx, g.edge_index)))
        h = F.relu(self.gn2(self.gc2(h, g.edge_index)))
        h = global_mean_pool(h, g.batch)
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

    def predict(self, data_loader):
        self.eval()

        with torch.no_grad():
            preds = [self(batch.cuda()) for batch in data_loader]

        return torch.vstack(preds)

    def emb(self, data_loader):
        embs = list()

        self.eval()
        with torch.no_grad():
            for b in data_loader:
                b = b.cuda()
                hx = self.fc1(b.x)
                h = F.relu(self.gn1(self.gc1(hx, b.edge_index, b.edge_attr)))
                h = F.relu(self.gn2(self.gc2(h, b.edge_index, b.edge_attr)))
                e = global_mean_pool(h, b.batch)
                embs.append(e)

        return torch.vstack(embs)


class GIN(nn.Module):
    def __init__(self, n_node_feats, dim_out):
        super(GIN, self).__init__()
        self.fc1 = nn.Linear(n_node_feats, 128)
        self.gc1 = GINConv(nn.Linear(128, 128))
        self.gn1 = LayerNorm(128)
        self.gc2 = GINConv(nn.Linear(128, 128))
        self.gn2 = LayerNorm(128)
        self.fc2 = nn.Linear(128, dim_out)

    def forward(self, g):
        hx = self.fc1(g.x)
        h = F.relu(self.gn1(self.gc1(hx, g.edge_index)))
        h = F.relu(self.gn2(self.gc2(h, g.edge_index)))
        h = global_mean_pool(h, g.batch)
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

    def predict(self, data_loader):
        self.eval()

        with torch.no_grad():
            preds = [self(batch.cuda()) for batch in data_loader]

        return torch.vstack(preds)

    def emb(self, data_loader):
        embs = list()

        self.eval()
        with torch.no_grad():
            for b in data_loader:
                b = b.cuda()
                hx = self.fc1(b.x)
                h = F.relu(self.gn1(self.gc1(hx, b.edge_index, b.edge_attr)))
                h = F.relu(self.gn2(self.gc2(h, b.edge_index, b.edge_attr)))
                e = global_mean_pool(h, b.batch)
                embs.append(e)

        return torch.vstack(embs)


class ECCNN(nn.Module):
    def __init__(self, n_node_feats, n_edge_feats, dim_out):
        super(ECCNN, self).__init__()
        self.fc1 = nn.Linear(n_node_feats, 64)
        self.efc1 = nn.Sequential(nn.Linear(n_edge_feats, 64), nn.ReLU(), nn.Linear(64, 64 * 64))
        self.gc1 = NNConv(64, 64, self.efc1)
        self.gn1 = LayerNorm(64)
        self.efc2 = nn.Sequential(nn.Linear(n_edge_feats, 64), nn.ReLU(), nn.Linear(64, 64 * 64))
        self.gc2 = NNConv(64, 64, self.efc2)
        self.gn2 = LayerNorm(64)
        self.fc2 = nn.Linear(64, dim_out)

    def forward(self, g):
        hx = self.fc1(g.x)
        h = F.relu(self.gn1(self.gc1(hx, g.edge_index, g.edge_attr)))
        h = F.relu(self.gn2(self.gc2(h, g.edge_index, g.edge_attr)))
        h = global_mean_pool(h, g.batch)
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

    def predict(self, data_loader):
        self.eval()

        with torch.no_grad():
            preds = [self(batch.cuda()) for batch in data_loader]

        return torch.vstack(preds)

    def emb(self, data_loader):
        embs = list()

        self.eval()
        with torch.no_grad():
            for b in data_loader:
                b = b.cuda()
                hx = self.fc1(b.x)
                h = F.relu(self.gn1(self.gc1(hx, b.edge_index, b.edge_attr)))
                h = F.relu(self.gn2(self.gc2(h, b.edge_index, b.edge_attr)))
                e = global_mean_pool(h, b.batch)
                embs.append(e)

        return torch.vstack(embs)


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
        h = global_mean_pool(h, g.batch)
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

    def predict(self, data_loader):
        self.eval()

        with torch.no_grad():
            preds = [self(batch.cuda()) for batch in data_loader]

        return torch.vstack(preds)

    def emb(self, data_loader):
        embs = list()

        self.eval()
        with torch.no_grad():
            for b in data_loader:
                b = b.cuda()
                hx = self.fc1(b.x)
                h = F.relu(self.gn1(self.gc1(hx, b.edge_index, b.edge_attr)))
                h = F.relu(self.gn2(self.gc2(h, b.edge_index, b.edge_attr)))
                e = global_mean_pool(h, b.batch)
                embs.append(e)

        return torch.vstack(embs)


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
        h = global_mean_pool(h, g.batch)
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

    def predict(self, data_loader):
        self.eval()

        with torch.no_grad():
            preds = [self(batch.cuda()) for batch in data_loader]

        return torch.vstack(preds)

    def emb(self, data_loader):
        embs = list()

        self.eval()
        with torch.no_grad():
            for b in data_loader:
                b = b.cuda()
                hx = self.fc1(b.x)
                h = F.relu(self.gn1(self.gc1(hx, b.edge_index, b.edge_attr)))
                h = F.relu(self.gn2(self.gc2(h, b.edge_index, b.edge_attr)))
                e = global_mean_pool(h, b.batch)
                embs.append(e)

        return torch.vstack(embs)


class SIGNNA(nn.Module):
    def __init__(self, gnn_c, gnn_e, dim_emb, dim_out):
        super(SIGNNA, self).__init__()
        self.gnn_c = nn.ModuleList(gnn_c) if isinstance(gnn_c, list) else nn.ModuleList([gnn_c])
        self.gnn_e = nn.ModuleList(gnn_e) if isinstance(gnn_e, list) else nn.ModuleList([gnn_e])
        self.fc1 = nn.Linear((len(self.gnn_c) + len(self.gnn_e)) * dim_emb, 64)
        self.fc2 = nn.Linear(64, dim_out)

    def forward(self, g_c, g_e):
        h_c = torch.hstack([(self.gnn_c[i](g_c[i])) for i in range(0, len(g_c))])
        h_e = torch.hstack([(self.gnn_e[i](g_e[i])) for i in range(0, len(g_e))])
        h = F.relu(self.fc1(torch.hstack([h_c, h_e])))
        out = self.fc2(h)

        return out

    def fit(self, data_loader, optimizer, criterion):
        self.train()
        train_loss = 0

        for g_react, g_env, y in data_loader:
            for g in g_react:
                g.cuda()

            for g in g_env:
                g.cuda()

            preds = self(g_react, g_env)
            loss = criterion(y.cuda(), preds)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.detach().item()

        return train_loss / len(data_loader)

    def predict(self, data_loader):
        self.eval()
        preds = list()

        with torch.no_grad():
            for g_react, g_env, y in data_loader:
                for g in g_react:
                    g.cuda()

                for g in g_env:
                    g.cuda()

                preds.append(self(g_react, g_env))

        return torch.vstack(preds)
