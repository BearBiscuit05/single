import copy
import os
import sys
import torch.nn as nn
import torch
import argparse
import time
import torch.distributed as dist
import torch.multiprocessing as mp
import argparse
import ast
import os.path as osp
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
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

@torch.no_grad()
def test(model,evaluator,data,subgraph_loader,split_idx):
    model.eval()

    out = model.inference(data.x,"cuda:0",subgraph_loader)

    y_true = data.y.cpu()
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': y_true[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    val_acc = evaluator.eval({
        'y_true': y_true[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y_true[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, val_acc, test_acc


def run(args,rank, world_size, dataset):
    
    data = dataset[0]
    data = data.to('cuda:0', 'x', 'y')
    if args.dataset == 'Reddit':
        train_idx = data.train_mask.nonzero(as_tuple=False).view(-1)
    elif args.dataset == 'ogb-products':
        train_idx = split_idx['train']
        val_idx = split_idx['valid']
        test_idx = split_idx['test']

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
    if args.dataset == 'Reddit':
        feat_size = 602
    elif args.dataset == 'ogb-products':
        feat_size = 100

    model = Net(feat_size, 256, 47,args.layers).to('cuda:0')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epochNum = 200
    epochTime = [0]
    testEpoch = [5,30,50,100,200]
    for epoch in range(1, epochNum+1):
        startTime = time.time()
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index.to('cuda:0'))[:batch.batch_size]
            loss = F.cross_entropy(out, batch.y[:batch.batch_size].squeeze())
            loss.backward()
            optimizer.step()
        eptime = time.time() - startTime
        totTime = epochTime[epoch-1] + eptime
        epochTime.append(totTime)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, time: {eptime:.5f}')

        if rank == 0 and epoch in testEpoch:  # We evaluate on a single GPU for now
            if args.dataset == 'Reddit':
                model.eval()
                with torch.no_grad():
                    out = model.inference(data.x, rank, subgraph_loader)
                res = out.argmax(dim=-1) == data.y.to(out.device)
                acc1 = int(res[data.train_mask].sum()) / int(data.train_mask.sum())
                acc2 = int(res[data.val_mask].sum()) / int(data.val_mask.sum())
                acc3 = int(res[data.test_mask].sum()) / int(data.test_mask.sum())
                print(f'Train: {acc1:.4f}, Val: {acc2:.4f}, Test: {acc3:.4f}')
            elif args.dataset == 'ogb-products':
                evaluator = Evaluator(name='ogbn-products')
                train_acc, val_acc, test_acc = test(model,evaluator,data,subgraph_loader,split_idx)
                print(f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, '
                            f'Test: {test_acc:.4f}')
    print("Average Training Time of {:d} Epoches:{:.6f}".format(epochNum,epochTime[epochNum]/epochNum))
    print("Total   Training Time of {:d} Epoches:{:.6f}".format(epochNum,epochTime[epochNum]))


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
        dataset = Reddit('../../../data/pyg_reddit')
        split_idx = None
    elif args.dataset == 'ogb-products':
        root = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'dataset')
        dataset = PygNodePropPredDataset('ogbn-products', root)
        split_idx = dataset.get_idx_split()

    run(args,0, world_size, dataset)
