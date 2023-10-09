import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
import dgl
import dgl.nn as dglnn
from dgl.data import AsNodePredDataset
from dgl.dataloading import DataLoader, NeighborSampler, MultiLayerFullNeighborSampler
from ogb.nodeproppred import DglNodePropPredDataset
from dgl_model import SAGE, GCN, GAT
import tqdm
import argparse
import ast
import sklearn.metrics
import numpy as np
import time
import sys

def train(args, device, g,model,train_idx,basicLoop=0,loop=10):
    sampler = NeighborSampler(args.fanout,  # fanout for [layer-0, layer-1, layer-2]
                            prefetch_node_feats=['feat'],
                            prefetch_labels=['label'])
    use_uva = (args.mode == 'mixed')
    train_dataloader = DataLoader(g, train_idx, sampler, device=device,
                                  batch_size=1024, shuffle=True,
                                  drop_last=False, num_workers=0,
                                  use_uva=use_uva)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    
    for epoch in range(loop):
        model.train()
        total_loss = 0
        startTime = time.time()
        count = 0
        for it, (_,_, blocks) in enumerate(train_dataloader):
            x = blocks[0].srcdata['feat']
            y = blocks[-1].dstdata['label']
            y_hat = model(blocks, x)
            y = y.to(torch.int64)
            loss = F.cross_entropy(y_hat, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            count = it
        trainTime = time.time()-startTime
        print("| Epoch {:05d} | Loss {:.4f} | Time {:.3f}s | Count {} |"
              .format(basicLoop+epoch, total_loss / (it+1), trainTime, count))

def load_dataset(dataset,path,featlen,mode=None):
    # 数据集的节点数
    if dataset == 'com_fr':
        nodenum = 65608366
    elif dataset == 'twitter':
        nodenum = 41652230
    elif dataset == 'uk-2007-05':
        nodenum = 105896555
    else:
        exit(-1)
    graphbin = "%s/%s/graph.bin" % (path,dataset)
    labelbin = "%s/%s/labels.bin" % (path,dataset) # 每个节点label 8字节
    featsbin = "%s/%s/feats_%d.bin" % (path,dataset,featlen)
    # 读取边集
    edges = np.fromfile(graphbin,dtype=np.int32)
    srcs = torch.tensor(edges[::2])
    dsts = torch.tensor(edges[1::2])
    # 读取特征
    feats = np.fromfile(featsbin,dtype=np.float32).reshape(nodenum,featlen)
    # label长度，comfr是8字节，其余4字节
    if dataset == 'com_fr':
        label = np.fromfile(labelbin,dtype=np.int32)
    elif dataset == 'twitter' or dataset == 'uk-2007-05':
        label = np.fromfile(labelbin,dtype=np.int64)
    # 构建dgl.Graph
    g = dgl.graph((srcs,dsts),num_nodes=nodenum,idtype=torch.int32)
    g.ndata['feat'] = torch.tensor(feats)
    g.ndata['label'] = torch.tensor(label)

    if mode == 'id_ordered' or mode == 'id_random':           # 以加载id二进制文件方法拿到训练节点
        trainbin = "%s/%s/train_%s.bin" % (path,dataset,mode)
        train_idx = np.fromfile(trainbin,dtype=np.int32)
    elif mode == 'mask':                                      # 以加载mask方法拿到训练节点
        trainbin = "%s/%s/train_mask.bin" % (path,dataset)
        trainmask = np.fromfile(trainbin,dtype=np.int32)
        train_idx = np.argwhere(trainmask > 0).squeeze()
    else:                                                     # 直接取1%作为训练节点
        trainnum = int(nodenum * 0.01)
        train_idx = np.arange(trainnum,dtype=np.int32)
    return g,train_idx


def gen_trainid_uk2007():
    trainid = np.arange(1059000,dtype=np.int32)
    trainid.tofile("/raid/bear/dataset/uk-2007-05/train_id_ordered.bin")
    random_numbers = np.random.choice(np.arange(0,105896555,dtype=np.int32), 1059000, replace=False)
    random_numbers.tofile("/raid/bear/dataset/uk-2007-05/train_id_random.bin")
    exit(0)

if __name__ == '__main__':
    #gen_trainid_uk2007()
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default='mixed', choices=['cpu', 'mixed', 'puregpu'],
                        help="Training mode. 'cpu' for CPU training, 'mixed' for CPU-GPU mixed training, "
                             "'puregpu' for pure-GPU training.")
    parser.add_argument('--fanout', type=ast.literal_eval, default=[10, 10, 10], help='Fanout value')
    parser.add_argument('--layers', type=int, default=3, help='Number of layers')
    parser.add_argument('--dataset', type=str, default='com_fr', help='Dataset name')
    parser.add_argument('--maxloop', type=int, default=20, help='max loop number')
    parser.add_argument('--model', type=str, default="SAGE", help='train model')
    args = parser.parse_args()
    if not torch.cuda.is_available():
        args.mode = 'cpu'
    print(f'Training in {args.mode} mode.')
    
    # load and preprocess dataset
    print('Loading data')
    datasetpath = "/raid/bear/dataset"
    if args.dataset == 'com_fr':
        g,train_idx = load_dataset(args.dataset,datasetpath,100,'id_ordered')
        out_size = 150
    elif args.dataset == 'twitter':
        g,train_idx = load_dataset(args.dataset,datasetpath,300,'id_ordered')
        out_size = 150
    elif args.dataset == 'uk-2007-05':
        g,train_idx = load_dataset(args.dataset,datasetpath,300)
        out_size = 150
    else:
        exit(0)
    
    g = g.to('cuda' if args.mode == 'puregpu' else 'cpu')
    device = torch.device('cpu' if args.mode == 'cpu' else 'cuda')
    
    # create GraphSAGE model

    in_size = g.ndata['feat'].shape[1]
    
    if args.model == "SAGE":
        model = SAGE(in_size, 256, out_size,args.layers).to(device)
    elif args.model == "GCN":
        model = GCN(in_size, 256, out_size,args.layers,F.relu,0.5).to(device)
    elif args.model == "GAT":
        model = GAT(in_size, 256, out_size,heads=[4,1]).to(device)
    else:
        # 如果 args.model 不是有效的模型选项，触发错误并退出程序
        print("Invalid model option. Please choose from 'SAGE', 'GCN', or 'GAT'.")
        sys.exit(1)  # 使用 sys.exit(1) 退出程序并返回错误状态码
    
    print('Training...')
    train(args,device,g,model,train_idx,basicLoop=0,loop=20)