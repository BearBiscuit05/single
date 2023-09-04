import copy
import os
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm
import argparse
import ast
import os.path as osp
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
from torch_geometric.datasets import Reddit
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv
import sys

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

def run(args, rank, world_size, dataset,split_idx=None):
    data = dataset[0]
    #data = data.to('cuda:0', 'x', 'y')  # Move to device for faster feature fetch.
    data = data.to('cuda:0', 'y')
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
        del subgraph_loader.data.x, subgraph_loader.data.y
        subgraph_loader.data.node_id = torch.arange(data.num_nodes)

    torch.manual_seed(12345)
    model = SAGE(dataset.num_features, 256, dataset.num_classes,args.layers).to('cuda:0')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochNum = 200
    epochTime = [0]
    testEpoch = [5,30,50,100,200]
    for epoch in range(1, epochNum+1):
        model.train()
        startTime = time.time()    
        for batch in train_loader:        
            optimizer.zero_grad()    
            out = model(batch.x.to('cuda:0'), batch.edge_index.to('cuda:0'))[:batch.batch_size]
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='pyg gcn program')
    parser.add_argument('--fanout', type=ast.literal_eval, default=[25, 10], help='Fanout value')
    parser.add_argument('--layers', type=int, default=2, help='Number of layers')
    parser.add_argument('--dataset', type=str, default='ogb-products', help='Dataset name')
    args = parser.parse_args()
    world_size = 1
    print('Let\'s use', world_size, 'GPUs!')
    
    # Print the parsed arguments
    print('Fanout:', args.fanout)
    print('Layers:', args.layers)
    print('Dataset:', args.dataset)

    # world_size = torch.cuda.device_count()
    if args.dataset == 'Reddit':
        dataset = Reddit('../../../data/pyg_reddit')
        run(args,0, world_size, dataset,split_idx=None)
    elif args.dataset == 'ogb-products':
        root = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'dataset')
        dataset = PygNodePropPredDataset('ogbn-products', root)
        split_idx = dataset.get_idx_split()
        evaluator = Evaluator(name='ogbn-products')
        run(args,0, world_size, dataset,split_idx)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    
    #mp.spawn(run, args=(world_size, dataset), nprocs=world_size, join=True)
