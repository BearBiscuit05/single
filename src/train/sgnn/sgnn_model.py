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

class DGL_SAGE(nn.Module):
    def __init__(self, in_size, hid_size, out_size,num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList()
        # three-layer GraphSAGE-mean
        self.layers.append(dglnn.SAGEConv(in_size, hid_size, 'mean'))
        for _ in range(num_layers - 2):
            self.layers.append(dglnn.SAGEConv(hid_size, hid_size, 'mean'))
        self.layers.append(dglnn.SAGEConv(hid_size, out_size, 'mean'))
        self.dropout = nn.Dropout(0.5)
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

        feat = feat.cpu()   # 防止显存空间不够
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
    
class DGL_GCN(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super(DGL_GCN, self).__init__()
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

class DGL_GAT(nn.Module):
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
    
from torch import Tensor
from torch_geometric.nn import SAGEConv,GATConv,GCNConv

class PYG_GCN(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int,
                 out_channels: int, num_layers: int = 2):
        super(PYG_GCN, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))
        self.num_layers = num_layers

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = x.relu_()
                x = F.dropout(x, p=0.5, training=self.training)
        return F.log_softmax(x, dim=1)

    def inference(self, x_all,subgraph_loader):
        pbar = tqdm.tqdm(total=x_all.size(0) * self.num_layers, position=0)
        pbar.set_description('Evaluating')
        for i in range(self.num_layers):
            xs = []
            for batch in subgraph_loader:
                x = x_all[batch.n_id].to("cuda:0")
                edge_index = batch.edge_index.to("cuda:0")
                x = self.convs[i](x, edge_index)
                x = x[:batch.batch_size]
                if i != self.num_layers - 1:
                    x = x.relu()
                xs.append(x.cpu())
                pbar.update(batch.batch_size)
            x_all = torch.cat(xs, dim=0)
        pbar.close()
        return x_all

class PYG_SAGE(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int,
                 out_channels: int, num_layers: int = 2):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))
        self.num_layers = num_layers

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = x.relu_()
                x = F.dropout(x, p=0.5, training=self.training)
        return x

    def inference(self, x_all,subgraph_loader):
        pbar = tqdm.tqdm(total=x_all.size(0) * self.num_layers, position=0)
        pbar.set_description('Evaluating')

        for i in range(self.num_layers):
            xs = []
            for batch in subgraph_loader:
                x = x_all[batch.n_id].to("cuda:0")
                edge_index = batch.edge_index.to("cuda:0")
                x = self.convs[i](x, edge_index)
                x = x[:batch.batch_size]
                if i != self.num_layers - 1:
                    x = x.relu()
                xs.append(x.cpu())
                pbar.update(batch.batch_size)
            x_all = torch.cat(xs, dim=0)
        pbar.close()
        return x_all

class PYG_GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads, num_layers=2):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.conv1 = GATConv(in_channels, hidden_channels, heads, dropout=0.6)
        # On the Pubmed dataset, use `heads` output heads in `conv2`.
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=8,
                             concat=False, dropout=0.6)
        self.convs.append(self.conv1)
        self.convs.append(self.conv2)
        self.num_layers = num_layers
        
    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return x

    def inference(self, x_all,subgraph_loader):
        pbar = tqdm.tqdm(total=x_all.size(0) * self.num_layers, position=0)
        pbar.set_description('Evaluating')

        for i in range(self.num_layers):
            xs = []
            for batch in subgraph_loader:
                x = x_all[batch.n_id].to("cuda:0")
                edge_index = batch.edge_index.to("cuda:0")
                x = self.convs[i](x, edge_index)
                x = x[:batch.batch_size]
                if i != self.num_layers - 1:
                    x = x.relu()
                xs.append(x.cpu())
                pbar.update(batch.batch_size)
            x_all = torch.cat(xs, dim=0)
        pbar.close()
        return x_all

