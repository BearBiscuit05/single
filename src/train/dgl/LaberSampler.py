import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
import dgl
import dgl.nn as dglnn
from dgl.data import AsNodePredDataset
from dgl.dataloading import DataLoader, NeighborSampler, MultiLayerFullNeighborSampler
from ogb.nodeproppred import DglNodePropPredDataset
import tqdm
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
import dgl
import dgl.nn as dglnn
from dgl.data import AsNodePredDataset
from ogb.nodeproppred import DglNodePropPredDataset
from dgl.nn.pytorch import GraphConv
import tqdm
import numpy as np
import time

class SAGE(nn.Module):
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

    def inference(self, g,device, batch_size):
        """Conduct layer-wise inference to get all the node embeddings."""
        feat = g.ndata['feat']
        sampler = MultiLayerFullNeighborSampler(1, prefetch_node_feats=['feat'])
        # sampler = NeighborSampler([15],  # fanout for [layer-0, layer-1, layer-2]
        #                     prefetch_node_feats=['feat'],
        #                     prefetch_labels=['label'])
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


dataset = AsNodePredDataset(DglNodePropPredDataset('ogbn-products',root="/home/bear/workspace/single-gnn/data/dataset"))
g = dataset[0]

model = SAGE(100, 256, 47,3).to("cuda:0")
train_idx = dataset.train_idx.to("cuda:0")

LSampler = dgl.dataloading.LaborSampler([50,50,50])
train_dataloader = DataLoader(g, train_idx, LSampler, device='cuda:0',
                                  batch_size=1024, shuffle=False,
                                  drop_last=False, num_workers=0,
                                  use_uva=True)

for it, (input_nodes, output_nodes, blocks) in enumerate(train_dataloader):
    print(blocks)
    if it == 10:
        break
print('-'*30)
sampler = NeighborSampler([100,100,100])
train_dataloader = DataLoader(g, train_idx, sampler, device='cuda:0',
                                  batch_size=1024, shuffle=False,
                                  drop_last=False, num_workers=0,
                                  use_uva=True)

for it, (input_nodes, output_nodes, blocks) in enumerate(train_dataloader):
    print(blocks)
    if it == 10:
        break