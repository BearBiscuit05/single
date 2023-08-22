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

def run(args,rank, world_size, dataset):
    
    data = dataset[0]
    data = data.to('cuda:0', 'x', 'y')
    train_idx = data.train_mask.nonzero(as_tuple=False).view(-1)

    kwargs = dict(batch_size=1024, num_workers=1, persistent_workers=True)
    train_loader = NeighborLoader(data, input_nodes=train_idx,
                                  num_neighbors=args.fanout, shuffle=True,
                                  drop_last=True, **kwargs)

    if rank == 0:  # Create single-hop evaluation neighbor loader:
        subgraph_loader = NeighborLoader(copy.copy(data), num_neighbors=[-1],
                                         shuffle=False, **kwargs)
        # No need to maintain these features during evaluation:
        del subgraph_loader.data.x, subgraph_loader.data.y
        # Add global node index information:
        subgraph_loader.data.node_id = torch.arange(data.num_nodes)

    torch.manual_seed(12345)
    model = Net(100, 256, 47,args.layers).to('cuda:0')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    count = 0
    for epoch in range(0, 11):
        startTime = time.time()
        model.train()
        for batch in train_loader:        
            optimizer.zero_grad()    
            out = model(batch.x, batch.edge_index.to('cuda:0'))[:batch.batch_size]
            loss = F.cross_entropy(out, batch.y[:batch.batch_size])
            loss.backward()
            optimizer.step()
        runTime = time.time() - startTime
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Time: {runTime:.3f}s')

    if rank == 0 and epoch % 5 == 0:  # We evaluate on a single GPU for now
        model.eval()
        with torch.no_grad():
            out = model.inference(data.x, rank, subgraph_loader)
        res = out.argmax(dim=-1) == data.y.to(out.device)
        acc1 = int(res[data.train_mask].sum()) / int(data.train_mask.sum())
        acc2 = int(res[data.val_mask].sum()) / int(data.val_mask.sum())
        acc3 = int(res[data.test_mask].sum()) / int(data.test_mask.sum())
        print(f'Train: {acc1:.4f}, Val: {acc2:.4f}, Test: {acc3:.4f}')


def collate_fn(data):
    """
    data 输入结构介绍：
        [graph,feat]
    """
    return data[0]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='pyg gcn program')
    parser.add_argument('--fanout', type=ast.literal_eval, default=[25, 10], help='Fanout value')
    parser.add_argument('--layers', type=int, default=2, help='Number of layers')
    parser.add_argument('--dataset', type=str, default='Reddit', help='Dataset name')
    args = parser.parse_args()

    world_size = 1
    print('Let\'s use', world_size, 'GPUs!')

    # Print the parsed arguments
    print('Fanout:', args.fanout)
    print('Layers:', args.layers)
    print('Dataset:', args.dataset)

    # Load the dataset based on the provided dataset name
    if args.dataset == 'Reddit':
        dataset = Reddit('.')
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    run(args,0, world_size, dataset)

