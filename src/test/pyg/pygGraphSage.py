import copy
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from torch_geometric.datasets import Reddit
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv
import sys

def genPYGBatchTemplate():
    zeros = torch.tensor([0])
    template_src = torch.empty(0,dtype=torch.int32)
    template_dst = torch.empty(0,dtype=torch.int32)
    batchsize = 4
    ptr = batchsize + 1
    fanout = [4, 3]
    
    seeds = [i for i in range(1, batchsize + 1)]
    for number in fanout:
        dst = copy.deepcopy(seeds)
        src = copy.deepcopy(seeds)
        for ids in seeds:
            for i in range(number-1):
                dst.append(ids)
                src.append(ptr)
                ptr += 1
        seeds = copy.deepcopy(src)
        template_src = torch.cat([template_src,torch.tensor(src,dtype=torch.int32)])
        template_dst = torch.cat([template_dst,torch.tensor(dst,dtype=torch.int32)])
    template_src = torch.cat([template_src,zeros])
    template_dst = torch.cat([template_dst,zeros])
    PYGTemplate = torch.stack([template_src,template_dst])
    return PYGTemplate




class SAGE(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int,
                 out_channels: int, num_layers: int = 2):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))
 
    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = x.relu_()
                x = F.dropout(x, p=0.5, training=self.training)
        return x

    @torch.no_grad()
    def inference(self, x_all: Tensor, device: torch.device,
                  subgraph_loader: NeighborLoader) -> Tensor:

        pbar = tqdm(total=len(subgraph_loader) * len(self.convs))
        pbar.set_description('Evaluating')

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch:
        for i, conv in enumerate(self.convs):
            xs = []
            for batch in subgraph_loader:
                x = x_all[batch.node_id.to(x_all.device)].to(device)
                x = conv(x, batch.edge_index.to(device))
                x = x[:batch.batch_size]
                if i < len(self.convs) - 1:
                    x = x.relu_()
                xs.append(x.cpu())
                pbar.update(1)
            x_all = torch.cat(xs, dim=0)

        pbar.close()
        return x_all

if __name__ == '__main__':
    PYGTemplate = genPYGBatchTemplate()
    print(PYGTemplate)
    features = torch.rand(PYGTemplate[0].shape[0], 100)
    model = SAGE(100, 256, 47).to('cuda:1')
    model.train()
    out = model(features.to('cuda:1'), PYGTemplate.to('cuda:1'))
    print(out.shape)