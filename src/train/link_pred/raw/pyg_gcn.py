# -*- coding:utf-8 -*-
import os
import sys
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv, GCNConv, SAGEConv
import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
import copy
import pickle
import random

import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
from torch.optim.lr_scheduler import StepLR
from torch_geometric.utils import negative_sampling
from tqdm import tqdm

root_path="."

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        # torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


class GCN_LP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN_LP, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def encode(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

    def decode(self, z, edge_label_index):
        src = z[edge_label_index[0]]
        dst = z[edge_label_index[1]]
        r = (src * dst).sum(dim=-1)
        return r

    def forward(self, data, edge_label_index):
        z = self.encode(data)
        return self.decode(z, edge_label_index)



def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_metrics(out, edge_label):
    edge_label = edge_label.cpu().numpy()
    out = out.cpu().numpy()
    pred = (out > 0.5).astype(int)
    auc = roc_auc_score(edge_label, out)
    f1 = f1_score(edge_label, pred)
    ap = average_precision_score(edge_label, out)

    return auc, f1, ap


def save_pickle(dataset, file_name):
    f = open(file_name, "wb")
    pickle.dump(dataset, f, protocol=4)
    f.close()


def load_pickle(file_name):
    f = open(file_name, "rb+")
    dataset = pickle.load(f)
    f.close()
    return dataset


def train_negative_sample(train_data):
    neg_edge_index = negative_sampling(
        edge_index=train_data.edge_index, num_nodes=train_data.num_nodes,
        num_neg_samples=train_data.edge_label_index.size(1), method='sparse')
    edge_label_index = torch.cat(
        [train_data.edge_label_index, neg_edge_index],
        dim=-1,
    )
    edge_label = torch.cat([
        train_data.edge_label,
        train_data.edge_label.new_zeros(neg_edge_index.size(1))
    ], dim=0)

    return edge_label, edge_label_index


@torch.no_grad()
def test(model, val_data, test_data):
    model.eval()
    # cal val loss
    criterion = torch.nn.BCEWithLogitsLoss().to(device)
    out = model(val_data,
                val_data.edge_label_index).view(-1)
    val_loss = criterion(out, val_data.edge_label)
    # cal metrics
    out = model(test_data,
                test_data.edge_label_index).view(-1).sigmoid()
    model.train()

    auc, f1, ap = get_metrics(out, test_data.edge_label)

    return val_loss, auc, ap


def train(model, train_data, val_data, test_data, save_model_path):
    model = model.to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
    criterion = torch.nn.BCEWithLogitsLoss().to(device)
    early_stopping = EarlyStopping(patience=50, verbose=True)
    scheduler = StepLR(optimizer, step_size=100, gamma=0.5)
    min_epochs = 10
    min_val_loss = np.Inf
    final_test_auc = 0
    final_test_ap = 0
    best_model = None
    model.train()
    for epoch in tqdm(range(100)):
        optimizer.zero_grad()
        edge_label, edge_label_index = train_negative_sample(train_data)
        out = model(train_data, edge_label_index).view(-1)
        loss = criterion(out, edge_label)
        loss.backward()
        optimizer.step()
        # validation
        val_loss, test_auc, test_ap = test(model, val_data, test_data)
        if epoch + 1 > min_epochs and val_loss < min_val_loss:
            min_val_loss = val_loss
            final_test_auc = test_auc
            final_test_ap = test_ap
            best_model = copy.deepcopy(model)
            # save model
            state = {'model': best_model.state_dict()}
            torch.save(state, save_model_path)

        # scheduler.step()
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        print('epoch {:03d} train_loss {:.8f} val_loss {:.4f} test_auc {:.4f} test_ap {:.4f}'
              .format(epoch, loss.item(), val_loss, test_auc, test_ap))

    state = {'model': best_model.state_dict()}
    torch.save(state, save_model_path)

    return final_test_auc, final_test_ap







def main(device,dataset,train_data, val_data, test_data):
    model = GCN_LP(dataset.num_features, 64, 128).to(device)
    test_auc, test_ap = train(model,
                              train_data,
                              val_data,
                              test_data,
                              save_model_path=root_path + '/models/sage.pkl')
    print('final best auc:', test_auc)
    print('final best ap:', test_ap)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = T.Compose([
        T.NormalizeFeatures(),
        T.ToDevice(device),
        T.RandomLinkSplit(num_val=0.1, num_test=0.1, is_undirected=True,
                        add_negative_train_samples=False, disjoint_train_ratio=0),
    ])

    dataset = Planetoid(root_path + '/data', name='Cora', transform=transform)
    train_data, val_data, test_data = dataset[0]

    print(train_data)
    print(val_data)
    print(test_data)
    main(device,dataset,train_data, val_data, test_data)
