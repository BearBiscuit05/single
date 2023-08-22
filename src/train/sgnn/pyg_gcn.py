import copy
import os
import sys
import torch.nn as nn
import torch
import time
import torch.distributed as dist
import torch.multiprocessing as mp
from torch_geometric.loader import NeighborLoader
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch_geometric.datasets import Reddit
from torch_geometric.nn import GCNConv
from torch import Tensor
current_folder = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_folder+"/../../"+"load")
from loader import CustomDataset

class Net(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int,
                 out_channels: int, num_layers: int = 2):
        super(Net, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = x.relu_()
                x = F.dropout(x, p=0.5, training=self.training)
        return F.log_softmax(x, dim=1)

        # x = self.conv1(x, edge_index)
        # x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        # x = self.conv2(x, edge_index)
        # return F.log_softmax(x, dim=1)

    @torch.no_grad()
    def inference(self, x_all: Tensor, device: torch.device,
                  subgraph_loader: NeighborLoader) -> Tensor:

        pbar = tqdm(total=len(subgraph_loader) * len(self.convs))
        pbar.set_description('Evaluating')
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

def run(rank, world_size, dataset):
    train_loader = DataLoader(dataset=dataset, batch_size=1024, collate_fn=collate_fn)#,pin_memory=True)
    torch.manual_seed(12345)
    model = Net(100, 256, 47).to('cuda:0')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    count = 0
    for epoch in range(0, dataset.epoch):
        startTime = time.time()
        model.train()
        for graph,feat,label,number in train_loader:
            optimizer.zero_grad()     
            out = model(feat.to('cuda:0'), graph)[:number]
            loss = F.cross_entropy(out, label[:number].to(torch.int64).to('cuda:0'))
            loss.backward()
            optimizer.step()
        runTime = time.time() - startTime
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Time: {runTime:.3f}s')

def collate_fn(data):
    """
    data 输入结构介绍：
        [graph,feat]
    """
    return data[0]

if __name__ == '__main__':
    # world_size = torch.cuda.device_count()
    world_size = 1
    print('Let\'s use', world_size, 'GPUs!')
    dataset = CustomDataset("./../../load/graphsage.json")
    run(0, world_size, dataset)
    
    #mp.spawn(run, args=(world_size, dataset), nprocs=world_size, join=True)
