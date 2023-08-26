import copy
import os
import os.path as osp
import sys
import torch.nn as nn
import torch
import time
import torch.distributed as dist
import torch.multiprocessing as mp
from torch_geometric.loader import NeighborLoader
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch_geometric.datasets import Reddit
from torch_geometric.nn import GCNConv
from torch import Tensor
current_folder = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_folder+"/../../"+"load")
from loader import CustomDataset
import argparse
import ast
import json

class Net(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int,
                 out_channels: int, num_layers: int = 2):
        super(Net, self).__init__()
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

        # x = self.conv1(x, edge_index)
        # x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        # x = self.conv2(x, edge_index)
        # return F.log_softmax(x, dim=1)

    def inference(self, x_all,subgraph_loader):
        pbar = tqdm(total=x_all.size(0) * self.num_layers)
        pbar.set_description('Evaluating')

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
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

@torch.no_grad()
def test(model,evaluator,data,subgraph_loader,split_idx):
    model.eval()

    out = model.inference(data.x,subgraph_loader)

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


def run(rank, model, world_size, dataset):
    train_loader = DataLoader(dataset=dataset, batch_size=dataset.batchsize, collate_fn=collate_fn)#,pin_memory=True)
    torch.manual_seed(12345)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(0, dataset.epoch):
        startTime = time.time()
        total_loss = 0
        model.train()
        for it,(graph,feat,label,number) in enumerate(train_loader):
            optimizer.zero_grad()     
            out = model(feat.to('cuda:0'), graph)[:number]
            loss = F.cross_entropy(out, label[:number].to(torch.int64).to('cuda:0'))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        runTime = time.time() - startTime
        print(f'Epoch: {epoch:02d}, Loss: {total_loss / (it+1):.4f}, Time: {runTime:.3f}s')


def collate_fn(data):
    """
    data 输入结构介绍：
        [graph,feat]
    """
    return data[0]

if __name__ == '__main__':
    # world_size = torch.cuda.device_count()
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default='mixed', choices=['cpu', 'mixed', 'puregpu'],
                        help="Training mode. 'cpu' for CPU training, 'mixed' for CPU-GPU mixed training, "
                             "'puregpu' for pure-GPU training.")
    # parser.add_argument('--fanout', type=ast.literal_eval, default=[25, 10], help='Fanout value')
    # parser.add_argument('--layers', type=int, default=2, help='Number of layers')
    # parser.add_argument('--dataset', type=str, default='Reddit', help='Dataset name')
    parser.add_argument('--json_path', type=str, default='.', help='Dataset name')
    args = parser.parse_args()

    data = None
    with open(args.json_path, 'r') as json_file:
        data = json.load(json_file)
    if data["dataset"] == "products_4":
        arg_dataset = 'ogb-products'
    elif data["dataset"] == "reddit_8":
        arg_dataset = 'Reddit'
    arg_fanout = data["fanout"]
    arg_layers = len(arg_fanout)

    world_size = 1
    print('Let\'s use', world_size, 'GPUs!')
    dataset = CustomDataset(args.json_path)    
    model = Net(data['featlen'], 256, data['classes'],arg_layers).to('cuda:0')
    run( 0, model ,world_size, dataset)
    
    if arg_dataset == 'ogb-products':
        root = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'dataset')
        dataset = PygNodePropPredDataset('ogbn-products', root)
        split_idx = dataset.get_idx_split()
        evaluator = Evaluator(name='ogbn-products')
        data = dataset[0].to("cuda:0", 'x', 'y')
        subgraph_loader = NeighborLoader(
            data,
            input_nodes=None,
            num_neighbors=[-1],
            batch_size=4096,
            num_workers=12,
            persistent_workers=True,
        )
        train_acc, val_acc, test_acc = test(model,evaluator,data,subgraph_loader,split_idx)
        print(f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, '
                    f'Test: {test_acc:.4f}')
    elif arg_dataset == 'Reddit':
        model.eval()
        dataset = Reddit('../../../data/pyg_reddit')
        data = dataset[0]
        data = data.to('cuda:0', 'x', 'y')
        subgraph_loader = NeighborLoader(
            data,
            input_nodes=None,
            num_neighbors=[-1],
            batch_size=4096,
            num_workers=12,
            persistent_workers=True,
        )
        with torch.no_grad():
            out = model.inference(data.x, subgraph_loader)
        res = out.argmax(dim=-1) == data.y.to(out.device)
        acc1 = int(res[data.train_mask].sum()) / int(data.train_mask.sum())
        acc2 = int(res[data.val_mask].sum()) / int(data.val_mask.sum())
        acc3 = int(res[data.test_mask].sum()) / int(data.test_mask.sum())
        print(f'Train: {acc1:.4f}, Val: {acc2:.4f}, Test: {acc3:.4f}')
