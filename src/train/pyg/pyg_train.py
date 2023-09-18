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
from pyg_model import SAGE, GCN, GAT



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

def run(args, dataset,split_idx=None):
    loopList = [0,10,20,30,50,100,150,200]
    data = dataset[0]
    data = data.to('cuda:0', 'x', 'y')  # Move to device for faster feature fetch.
    # data = data.to('cuda:0', 'y')
    if args.dataset == 'Reddit':
        train_idx = data.train_mask.nonzero(as_tuple=False).view(-1)
    elif args.dataset == 'ogb-products':
        train_idx = split_idx['train']
        val_idx = split_idx['valid']
        test_idx = split_idx['test']

    kwargs = dict(batch_size=1024, num_workers=1, persistent_workers=True)
    # train_loader = NeighborLoader(data, input_nodes=train_idx,
    #                               num_neighbors=args.fanout, shuffle=True,
    #                               drop_last=True, **kwargs)

    train_loader = NeighborLoader(
        data,
        input_nodes=train_idx,
        num_neighbors=args.fanout,
        batch_size=1024,
        shuffle=True,
        num_workers=12,
        persistent_workers=True,
    )
    subgraph_loader = NeighborLoader(copy.copy(data), num_neighbors=[-1],
                                        shuffle=False, **kwargs)
    del subgraph_loader.data.x, subgraph_loader.data.y
    subgraph_loader.data.node_id = torch.arange(data.num_nodes)

    torch.manual_seed(12345)
    if args.dataset == 'Reddit':
        feat_size,classNUM = 602,41
    elif args.dataset == 'ogb-products':
        feat_size,classNUM = 100,47

    if args.model == "SAGE":
        model = SAGE(feat_size, 256, classNUM,args.layers).to('cuda:0')
    elif args.model == "GCN":
        model = GCN(feat_size, 256,classNUM,args.layers).to('cuda:0')
    elif args.model == "GAT":
        model = GAT(feat_size, 256, classNUM, 4).to('cuda:0')
    else:
        print("Invalid model option. Please choose from 'SAGE', 'GCN', or 'GAT'.")
        sys.exit(1)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for index in range(1,len(loopList)):
        if loopList[index] > args.maxloop:
            break
        _loop = loopList[index] - loopList[index - 1]
        basicLoop = loopList[index - 1]
        for epoch in range(_loop):
            model.train()
            startTime = time.time() 
            total_loss = 0   
            count = 0
            for it, batch in enumerate(train_loader):        
                optimizer.zero_grad()    
                out = model(batch.x.to('cuda:0'), batch.edge_index.to('cuda:0'))[:batch.batch_size]
                loss = F.cross_entropy(out, batch.y[:batch.batch_size].squeeze().to('cuda:0'))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                count = it
            trainTime = time.time() - startTime
            print("| Epoch {:05d} | Loss {:.4f} | Time {:.3f}s | Count {} |"
              .format(basicLoop+epoch, total_loss / (it+1), trainTime, count))

            # if (epoch+1) in loopList :  # We evaluate on a single GPU for now
            #     if args.dataset == 'Reddit':
            #         model.eval()
            #         with torch.no_grad():
            #             out = model.inference(data.x, "cuda:0", subgraph_loader)
            #         res = out.argmax(dim=-1) == data.y.to(out.device)
            #         acc1 = int(res[data.train_mask].sum()) / int(data.train_mask.sum())
            #         acc2 = int(res[data.val_mask].sum()) / int(data.val_mask.sum())
            #         acc3 = int(res[data.test_mask].sum()) / int(data.test_mask.sum())
            #         print(f'Train: {acc1:.4f}, Val: {acc2:.4f}, Test: {acc3:.4f}')
            #     elif args.dataset == 'ogb-products':
            #         evaluator = Evaluator(name='ogbn-products')
            #         train_acc, val_acc, test_acc = test(model,evaluator,data,subgraph_loader,split_idx)
            #         print(f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, '
            #                     f'Test: {test_acc:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='pyg gcn program')
    parser.add_argument('--fanout', type=ast.literal_eval, default=[25, 10], help='Fanout value')
    parser.add_argument('--layers', type=int, default=2, help='Number of layers')
    parser.add_argument('--dataset', type=str, default='ogb-products', help='Dataset name')
    parser.add_argument('--maxloop', type=int, default=10, help='max loop number')
    parser.add_argument('--model', type=str, default="SAGE", help='train model')

    args = parser.parse_args()
    world_size = 1
    print('Let\'s use', world_size, 'GPUs!')
    print('Fanout:', args.fanout)
    print('Layers:', args.layers)
    print('Dataset:', args.dataset)

    # world_size = torch.cuda.device_count()
    if args.dataset == 'Reddit':
        dataset = Reddit('/home/bear/workspace/singleGNN/data/reddit/pyg_reddit')
        run(args, dataset,split_idx=None)
    elif args.dataset == 'ogb-products':
        root = osp.join(osp.dirname(osp.realpath(__file__)), '/home/bear/workspace/singleGNN/data/', 'dataset')
        dataset = PygNodePropPredDataset('ogbn-products', root)
        split_idx = dataset.get_idx_split()
        evaluator = Evaluator(name='ogbn-products')
        run(args, dataset,split_idx)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
