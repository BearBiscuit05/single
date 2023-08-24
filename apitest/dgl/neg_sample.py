import argparse
from ogb.nodeproppred import DglNodePropPredDataset
import dgl
import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from dgl.dataloading import (
    as_edge_prediction_sampler,
    DataLoader,
    MultiLayerFullNeighborSampler,
    negative_sampler,
    NeighborSampler,
)
from ogb.linkproppred import DglLinkPropPredDataset, Evaluator
from dgl.data import AsNodePredDataset

dataset = AsNodePredDataset(DglNodePropPredDataset('ogbn-products'))
g = dataset[0]

sampler = NeighborSampler([3], prefetch_node_feats=["feat"])
sampler = as_edge_prediction_sampler(
    sampler,
    negative_sampler=negative_sampler.Uniform(1),
)
device='cuda:0'
seed_edges = torch.Tensor([0,2]).to(torch.int64)
use_uva = True
print(g.edges())
dataloader = DataLoader(
    g,
    seed_edges,
    sampler,
    device=device,
    batch_size=2,
    shuffle=True,
    drop_last=False,
    num_workers=0,
    use_uva=use_uva,
)

for it, (input_nodes, pair_graph, neg_pair_graph, blocks) in enumerate(
            dataloader):
    print("input:",input_nodes)
    print("pair_graph:",pair_graph.ndata[dgl.NID])
    print("pair_graph edges:",pair_graph.edges())
    print("neg_pair_graph:",neg_pair_graph.ndata[dgl.NID])
    print("neg_pair_graph edges:",neg_pair_graph.edges())
