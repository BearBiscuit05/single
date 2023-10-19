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
import numpy as np
from torch_geometric.data import Data

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
    data.y = data.y.to(torch.int64)
    #data = data.to('cuda:0', 'x', 'y')  # Move to device for faster feature fetch.
    data = data.to('cuda:0', 'y')
    if args.dataset == 'Reddit':
        train_idx = data.train_mask.nonzero(as_tuple=False).view(-1)
    elif args.dataset == 'ogb-products' or args.dataset == 'ogb-papers100M':
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
    elif args.dataset == 'ogb-papers100M':
        feat_size,classNUM = 128,172

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

            if (epoch+1) in loopList :  # We evaluate on a single GPU for now
                if args.dataset == 'Reddit':
                    model.eval()
                    with torch.no_grad():
                        out = model.inference(data.x, "cuda:0", subgraph_loader)
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


def testRun(args,Gdata,trainIDs):
    # data.x feat
    # data.y label
    print("...")
    torch.manual_seed(12345)
    classNUM = 150
    feat_size = Gdata.x.shape[1]
    if args.model == "SAGE":
        model = SAGE(feat_size, 256, classNUM,args.layers).to('cuda:0')
    elif args.model == "GCN":
        model = GCN(feat_size, 256,classNUM,args.layers).to('cuda:0')
    elif args.model == "GAT":
        model = GAT(feat_size, 256, classNUM, 4).to('cuda:0')
    else:
        print("Invalid model option. Please choose from 'SAGE', 'GCN', or 'GAT'.")
        sys.exit(1)
    print("...")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    trainIDs = trainIDs[:1024]
    Gdata.y = Gdata.y.to(torch.int64)
    print("...")
    train_loader = NeighborLoader(
        Gdata,
        input_nodes=trainIDs,
        num_neighbors=args.fanout,
        batch_size=1024,
        shuffle=True,
        num_workers=12,
        persistent_workers=True,
    )
    print("...")
    for epoch in range(10):
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
            .format(epoch, total_loss / (it+1), trainTime, count))

def load_dataset(dataset,path,featlen,mode=None):
    # 数据集的节点数
    # if dataset == 'com_fr':
    #     nodenum = 65608366
    # elif dataset == 'twitter':
    #     nodenum = 41652230
    # elif dataset == 'uk-2006-05':
    #     nodenum = 105896555
    # else:
    #     exit(-1)
    graphbin = "%s/%s/graph.bin" % (path,dataset)
    labelbin = "%s/%s/labels.bin" % (path,dataset) # 每个节点label 8字节
    featsbin = "%s/%s/feats_%d.bin" % (path,dataset,featlen)
    # 读取边集
    edges = np.fromfile(graphbin,dtype=np.int32)
    srcs = torch.tensor(edges[::2]).to(torch.int64)
    dsts = torch.tensor(edges[1::2]).to(torch.int64)
    # 读取特征
    feats = np.fromfile(featsbin,dtype=np.float32).reshape(-1,featlen)
    feats = torch.Tensor(feats)
    # label长度，comfr是8字节，其余4字节
    if dataset == 'com_fr':
        label = np.fromfile(labelbin,dtype=np.int32)
    elif dataset == 'twitter' or dataset == 'uk-2006-05':
        label = np.fromfile(labelbin,dtype=np.int64)
    label = torch.Tensor(label).to(torch.int64)
    edgeList = torch.stack((srcs,dsts),dim=0)
    data = Data(x=feats, edge_index=edgeList, y=label)
    
    if mode == 'id_ordered' or mode == 'id_random':           # 以加载id二进制文件方法拿到训练节点
        trainbin = "%s/%s/train_%s.bin" % (path,dataset,mode)
        train_idx = np.fromfile(trainbin,dtype=np.int32)
    elif mode == 'mask':                                      # 以加载mask方法拿到训练节点
        trainbin = "%s/%s/train_mask.bin" % (path,dataset)
        trainmask = np.fromfile(trainbin,dtype=np.int32)
        train_idx = np.argwhere(trainmask > 0).squeeze()
    else:                                                     # 直接取1%作为训练节点
        trainnum = int(len(srcs) * 0.01)
        train_idx = np.arange(trainnum,dtype=np.int64)
        # train_idx = srcs[:trainnum]
    return data,train_idx


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='pyg gcn program')
    parser.add_argument('--fanout', type=ast.literal_eval, default=[10, 10, 10], help='Fanout value')
    parser.add_argument('--layers', type=int, default=3, help='Number of layers')
    parser.add_argument('--dataset', type=str, default='uk-2006-05', help='Dataset name')
    parser.add_argument('--maxloop', type=int, default=10, help='max loop number')
    parser.add_argument('--model', type=str, default="SAGE", help='train model')

    args = parser.parse_args()
    world_size = 1
    print('Let\'s use', world_size, 'GPUs!')
    print('Fanout:', args.fanout)
    print('Layers:', args.layers)
    print('Dataset:', args.dataset)

    datasetpath = "/raid/bear/dataset"

    if args.dataset == 'Reddit':
        dataset = Reddit('/home/bear/workspace/singleGNN/data/reddit/pyg_reddit')
        run(args, dataset,split_idx=None)
    elif args.dataset == 'ogb-products':
        root = osp.join(osp.dirname(osp.realpath(__file__)), '/home/bear/workspace/singleGNN/data/', 'dataset')
        dataset = PygNodePropPredDataset('ogbn-products', root)
        split_idx = dataset.get_idx_split()
        evaluator = Evaluator(name='ogbn-products')
        run(args, dataset,split_idx)
    elif args.dataset == 'ogb-papers100M':
        root = "/raid/bear/wget_paper"
        print("root:",root)
        dataset = PygNodePropPredDataset('ogbn-papers100M', root)
        split_idx = dataset.get_idx_split()
        print("加载完毕")
        # evaluator = Evaluator(name='ogbn-papers100M')
        run(args, dataset,split_idx)
    elif args.dataset == 'com_fr':
        Gdata,train_idx = load_dataset(args.dataset,datasetpath,100,'id_ordered')
        out_size = 150
        testRun(args,Gdata,train_idx)
    elif args.dataset == 'twitter':
        Gdata,train_idx = load_dataset(args.dataset,datasetpath,300,'id_ordered')
        out_size = 150
        testRun(args,Gdata,train_idx)
    elif args.dataset == 'uk-2006-05':
        Gdata,train_idx = load_dataset(args.dataset,datasetpath,100)
        out_size = 150
        testRun(args,Gdata,train_idx)
    else:
        print("dataset name error....")
        exit(0)
    # else:
    #     raise ValueError(f"Unsupported dataset: {args.dataset}")
