import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
import dgl
import dgl.nn as dglnn
from dgl.data import AsNodePredDataset
from dgl.dataloading import DataLoader, NeighborSampler, MultiLayerFullNeighborSampler
from ogb.nodeproppred import DglNodePropPredDataset
from dgl.nn.pytorch import GraphConv
import tqdm
import argparse
import ast
import sklearn.metrics
import numpy as np
import time
import sys


class MLP(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 embed_dim,
                 num_layers: int,
                 act: str = 'ReLU',
                 bn: bool = False,
                 end_up_with_fc=False,
                 bias=True):
        super(MLP, self).__init__()
        self.module_list = []
        for i in range(num_layers):
            d_in = input_dim if i == 0 else hidden_dim
            d_out = embed_dim if i == num_layers - 1 else hidden_dim
            self.module_list.append(nn.Linear(d_in, d_out, bias=bias))
            if end_up_with_fc:
                continue
            if bn:
                self.module_list.append(nn.BatchNorm1d(d_out))
            self.module_list.append(getattr(nn, act)(True))
        self.module_list = nn.Sequential(*self.module_list)

    def forward(self, x):
        return self.module_list(x)


class ACC_SAGE(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.res_linears = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, 'mean', bias=False, feat_drop=dropout))
        self.bns.append(torch.nn.BatchNorm1d(n_hidden))
        self.res_linears.append(torch.nn.Linear(in_feats, n_hidden))
        for i in range(1, n_layers - 1):
            self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, 'mean', bias=False, feat_drop=dropout))
            self.bns.append(torch.nn.BatchNorm1d(n_hidden))
            self.res_linears.append(torch.nn.Identity())
        self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, 'mean', bias=False, feat_drop=dropout))
        self.bns.append(torch.nn.BatchNorm1d(n_hidden))
        self.res_linears.append(torch.nn.Identity())
        self.mlp = MLP(in_feats + n_hidden * n_layers, 2 * n_classes, n_classes, num_layers=2, bn=True,
                       end_up_with_fc=True, act='LeakyReLU')
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.profile = locals()

    def forward(self, blocks):
        collect = []
        h = blocks[0].srcdata['feat']
        h = self.dropout(h)
        num_output_nodes = blocks[-1].num_dst_nodes()
        collect.append(h[:num_output_nodes])
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h_res = h[:block.num_dst_nodes()]
            h = layer(block, h)
            h = self.bns[l](h)
            h = self.activation(h)
            h = self.dropout(h)
            collect.append(h[:num_output_nodes])
            h += self.res_linears[l](h_res)
        return self.mlp(torch.cat(collect, -1))


class SAGE(nn.Module):
    def __init__(self, in_size, hid_size, out_size,num_layers=2,dropout=0.5):
        super().__init__()
        self.layers = nn.ModuleList()
        # three-layer GraphSAGE-mean
        self.layers.append(dglnn.SAGEConv(in_size, hid_size, 'mean'))
        for _ in range(num_layers - 2):
            self.layers.append(dglnn.SAGEConv(hid_size, hid_size, 'mean'))
        self.layers.append(dglnn.SAGEConv(hid_size, out_size, 'mean'))
        self.dropout = nn.Dropout(dropout)
        self.hid_size = hid_size
        self.out_size = out_size

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h

    def inference(self, g, device, batch_size):
        """Conduct layer-wise inference to get all the node embeddings."""
        feat = g.ndata['feat']
        sampler = MultiLayerFullNeighborSampler(1, prefetch_node_feats=['feat'])
        dataloader = DataLoader(
                g, torch.arange(g.num_nodes()).to(g.device), sampler, device=device,
                batch_size=batch_size, shuffle=False, drop_last=False,
                num_workers=0)
        buffer_device = torch.device('cpu')
        pin_memory = (buffer_device != device)

        feat = feat.cpu()
        for l, layer in enumerate(self.layers):
            y = torch.empty(
                g.num_nodes(), self.hid_size if l != len(self.layers) - 1 else self.out_size,
                device=buffer_device, pin_memory=pin_memory)
            
            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader, position=0):
                x = feat[input_nodes.to(feat.device)].to(device)
                h = layer(blocks[0], x) # len(blocks) = 1
                if l != len(self.layers) - 1:
                    h = F.relu(h)
                    h = self.dropout(h)
                # by design, our output nodes are contiguous
                y[output_nodes[0]:output_nodes[-1]+1] = h.to(buffer_device)
            feat = y
        return y
    
class GCN(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        self.hid_size = n_hidden
        self.out_size = n_classes
        self.layers.append(GraphConv(in_feats, n_hidden, activation=activation, allow_zero_in_degree=True))
        for _ in range(1, n_layers - 1):
            self.layers.append(
                GraphConv(n_hidden, n_hidden, activation=activation, allow_zero_in_degree=True))
        self.layers.append(
            GraphConv(n_hidden, n_classes, allow_zero_in_degree=True))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, blocks, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(blocks[i], h)
        return h

    def inference(self, g, device, batch_size):
        """Conduct layer-wise inference to get all the node embeddings."""
        feat = g.ndata['feat']
        sampler = MultiLayerFullNeighborSampler(1, prefetch_node_feats=['feat'])
        dataloader = DataLoader(
                g, torch.arange(g.num_nodes()).to(g.device), sampler, device=device,
                batch_size=batch_size, shuffle=False, drop_last=False,
                num_workers=0)
        buffer_device = torch.device('cpu')
        pin_memory = (buffer_device != device)

        for l, layer in enumerate(self.layers):
            y = torch.empty(
                g.num_nodes(), self.hid_size if l != len(self.layers) - 1 else self.out_size,
                device=buffer_device, pin_memory=pin_memory)
            feat = feat.to(device)
            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader, position=0):
                x = feat[input_nodes]
                h = layer(blocks[0], x) # len(blocks) = 1
                if l != len(self.layers) - 1:
                    h = F.relu(h)
                    h = self.dropout(h)
                # by design, our output nodes are contiguous
                y[output_nodes[0]:output_nodes[-1]+1] = h.to(buffer_device)
            feat = y
        return y

class GAT(nn.Module):
    def __init__(self,in_size, hid_size, out_size, heads):
        super().__init__()
        self.layers = nn.ModuleList()
        # two-layer GAT
        self.heads = heads
        self.layers.append(dglnn.GATConv(in_size, hid_size, heads[0], feat_drop=0.6, attn_drop=0.6, activation=F.elu,allow_zero_in_degree=True))
        self.layers.append(dglnn.GATConv(hid_size*heads[0], out_size, heads[1], feat_drop=0.6, attn_drop=0.6, activation=None,allow_zero_in_degree=True))
        self.hid_size = hid_size
        self.out_size = out_size

    def forward(self, g, inputs):
        h = inputs
        for i, layer in enumerate(self.layers):
            h = layer(g[i], h)
            if i == 1:  # last layer 
                h = h.mean(1)
            else:       # other layer(s)
                h = h.flatten(1)
        return h

    def inference(self, g, device, batch_size):
        """Conduct layer-wise inference to get all the node embeddings."""
        feat = g.ndata['feat']
        sampler = MultiLayerFullNeighborSampler(1, prefetch_node_feats=['feat'])
        dataloader = DataLoader(
                g, torch.arange(g.num_nodes()).to(g.device), sampler, device=device,
                batch_size=batch_size, shuffle=False, drop_last=False,
                num_workers=0)
        buffer_device = torch.device('cpu')
        pin_memory = (buffer_device != device)

        for l, layer in enumerate(self.layers):
            y = torch.empty(
                g.num_nodes(), self.hid_size*self.heads[0] if l != len(self.layers) - 1 else self.out_size,
                device=buffer_device, pin_memory=pin_memory)
            feat = feat.to(device)
            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader, position=0):
                x = feat[input_nodes]
                h = layer(blocks[0], x) # len(blocks) = 1
                if l == 1:  # last layer 
                    h = h.mean(1)
                else:       # other layer(s)
                    h = h.flatten(1)
                # by design, our output nodes are contiguous
                y[output_nodes[0]:output_nodes[-1]+1] = h.to(buffer_device)
            feat = y
        return y


def count_parameters(model):
    return sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad])


if __name__ == '__main__':
    sage = ACC_SAGE(128, 512, 172, 3, torch.nn.functional.relu, 0)
    print(count_parameters(sage))