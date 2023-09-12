import copy
import os
import os.path as osp
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch_geometric.datasets import Reddit
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv
import sys
import time
current_folder = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_folder+"/../../"+"load")
from loader import CustomDataset
import argparse
import ast
import json
from sgnn_model import PYG_SAGE,PYG_GCN,PYG_GAT


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

def run_test(arg_dataset,model):
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
            num_workers=8,#此处A100可设置12，V100设置为8否则warning
            persistent_workers=True,
        )
        begTime = time.time()
        train_acc, val_acc, test_acc = test(model,evaluator,data,subgraph_loader,split_idx)
        endTime = time.time()
        print(f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')
        print('Test Time:',endTime-begTime)
    elif arg_dataset == 'Reddit':
        model.eval()
        dataset = Reddit('../../../data/reddit/pyg_reddit')
        data = dataset[0]
        data = data.to('cuda:0', 'x', 'y')
        subgraph_loader = NeighborLoader(
            data,
            input_nodes=None,
            num_neighbors=[-1],
            batch_size=4096,
            num_workers=8,#此处A100可设置12，V100设置为8否则warning
            persistent_workers=True,
        )
        begTime = time.time()
        with torch.no_grad():
            out = model.inference(data.x, subgraph_loader)
        endTime = time.time()
        res = out.argmax(dim=-1) == data.y.to(out.device)
        acc1 = int(res[data.train_mask].sum()) / int(data.train_mask.sum())
        acc2 = int(res[data.val_mask].sum()) / int(data.val_mask.sum())
        acc3 = int(res[data.test_mask].sum()) / int(data.test_mask.sum())
        print(f'Train: {acc1:.4f}, Val: {acc2:.4f}, Test: {acc3:.4f}')
        print('Test Time:',endTime-begTime)
    
def run(arg_dataset,rank,model,world_size,dataset):
    train_loader = DataLoader(dataset=dataset, batch_size=dataset.batchsize, collate_fn=collate_fn)#,pin_memory=True)
    torch.manual_seed(12345)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epochTime = [0]
    testEpoch = [5,30,50,100,200]
    for epoch in range(1,dataset.epoch+1):
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
        eptime = time.time() - startTime
        totTime = epochTime[epoch-1] + eptime
        epochTime.append(totTime)
        print(f'Epoch: {epoch:03d}, Loss: {total_loss / (it+1):.4f}, Time: {eptime:.6f}s')
        if rank == 0 and epoch in testEpoch:
            run_test(arg_dataset,model)
    print("Average Training Time of {:d} Epoches:{:.6f}".format(dataset.epoch,epochTime[dataset.epoch]/dataset.epoch))
    print("Total   Training Time of {:d} Epoches:{:.6f}".format(dataset.epoch,epochTime[dataset.epoch]))


def collate_fn(data):
    return data[0]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default='mixed', choices=['cpu', 'mixed', 'puregpu'],
                        help="Training mode. 'cpu' for CPU training, 'mixed' for CPU-GPU mixed training, "
                             "'puregpu' for pure-GPU training.")
    #parser.add_argument('--fanout', type=ast.literal_eval, default=[25, 10], help='Fanout value')
    #parser.add_argument('--layers', type=int, default=2, help='Number of layers')
    #parser.add_argument('--dataset', type=str, default='Reddit', help='Dataset name')
    parser.add_argument('--json_path', type=str, default='.', help='Dataset name')
    args = parser.parse_args()


    data = None
    with open(args.json_path, 'r') as json_file:
        data = json.load(json_file)
    
    print('Loading data')
    if data["dataset"] == "products_4":
        arg_dataset = 'ogb-products'
    elif data["dataset"] == "reddit_8":
        arg_dataset = 'Reddit'
    arg_fanout = data["fanout"]
    arg_layers = len(arg_fanout)

    dataset = CustomDataset(args.json_path)
    if data['model'] == "SAGE":
        model = PYG_SAGE(data['featlen'], 256, data['classes'],arg_layers).to('cuda:0')
    elif data['model'] == "GCN":
        model = PYG_GCN(data['featlen'], 256, data['classes'],arg_layers).to('cuda:0')
    elif data["model"] == "GAT":
        model = PYG_GAT(data['featlen'], 256, data['classes'], 4).to('cuda:0')
    else:
        print("Invalid model option. Please choose from 'SAGE', 'GCN', or 'GAT'.")
        sys.exit(1)
    
    run(model,dataset)
    if arg_dataset == 'ogb-products':
        root = osp.join(osp.dirname(osp.realpath(__file__)), '/home/bear/workspace/singleGNN/data/', 'dataset')
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
        dataset = Reddit('/home/bear/workspace/singleGNN/data/reddit/pyg_reddit')
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
