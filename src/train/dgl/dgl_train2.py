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

def load_comfr():
    nodenum = 65608366
    featlen = 100
    graphbin = "/raid/bear/dataset/com_fr/graph.bin"
    labelbin = "/raid/bear/dataset/com_fr/labels.bin" # 每个节点label 4字节
    featsbin = "/raid/bear/dataset/com_fr/feats_%d.bin" % featlen
    trainbin = "/raid/bear/dataset/com_fr/train_id_ordered.bin"
    edges = np.fromfile(graphbin,dtype=np.int32)
    srcs = torch.tensor(edges[::2])
    dsts = torch.tensor(edges[1::2])
    feats = np.fromfile(featsbin,dtype=np.float32).reshape(nodenum,featlen)
    label = np.fromfile(labelbin,dtype=np.int32)
    g = dgl.graph((srcs,dsts),num_nodes=nodenum,idtype=torch.int32)
    g.ndata['feat'] = torch.tensor(feats)
    g.ndata['label'] = torch.tensor(label)
    train_idx = np.fromfile(trainbin,dtype=np.int32)
    return g,train_idx

def load_twitter():
    nodenum = 41652230
    featlen = 300
    graphbin = "/raid/bear/dataset/twitter/graph.bin"
    labelbin = "/raid/bear/dataset/twitter/labels.bin" # 每个节点label 8字节
    featsbin = "/raid/bear/dataset/twitter/feats_%d.bin" % featlen
    trainbin = "/raid/bear/dataset/twitter/train_id_ordered.bin"
    edges = np.fromfile(graphbin,dtype=np.int32)
    srcs = torch.tensor(edges[::2])
    dsts = torch.tensor(edges[1::2])
    feats = np.fromfile(featsbin,dtype=np.float32).reshape(nodenum,featlen)
    label = np.fromfile(labelbin,dtype=np.int64)
    g = dgl.graph((srcs,dsts),num_nodes=nodenum,idtype=torch.int32)
    g.ndata['feat'] = torch.tensor(feats)
    g.ndata['label'] = torch.tensor(label)
    train_idx = np.fromfile(trainbin,dtype=np.int32)
    return g,train_idx

def load_uk2007():
    nodenum = 105896555
    featlen = 300
    graphbin = "/raid/bear/dataset/uk-2007-05/graph.bin"
    labelbin = "/raid/bear/dataset/uk-2007-05/labels.bin" # 每个节点label 8字节
    featsbin = "/raid/bear/dataset/uk-2007-05/feats_%d.bin" % featlen
    #trainbin = "/raid/bear/dataset/uk-2007-05/train_id_ordered.bin"
    edges = np.fromfile(graphbin,dtype=np.int32)
    srcs = torch.tensor(edges[::2])
    dsts = torch.tensor(edges[1::2])
    feats = np.fromfile(featsbin,dtype=np.float32).reshape(nodenum,featlen)
    # 测试用，裁减到100维
    feats = feats[:,0:100]
    print(feats.shape)
    label = np.fromfile(labelbin,dtype=np.int64)
    g = dgl.graph((srcs,dsts),num_nodes=nodenum,idtype=torch.int32)
    g.ndata['feat'] = torch.tensor(feats)
    g.ndata['label'] = torch.tensor(label)
    #train_idx = np.fromfile(trainbin,dtype=np.int32)
    train_idx = np.arange(1059000,dtype=np.int32)
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
    parser.add_argument('--dataset', type=str, default='com-fr', help='Dataset name')
    parser.add_argument('--maxloop', type=int, default=20, help='max loop number')
    parser.add_argument('--model', type=str, default="SAGE", help='train model')
    args = parser.parse_args()
    if not torch.cuda.is_available():
        args.mode = 'cpu'
    print(f'Training in {args.mode} mode.')
    
    # load and preprocess dataset
    print('Loading data')
    if args.dataset == 'com-fr':
        g,train_idx = load_comfr()
        out_size = 150
    elif args.dataset == 'twitter':
        g,train_idx = load_twitter()
        out_size = 150
    elif args.dataset == 'uk-2007':
        g,train_idx = load_uk2007()
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