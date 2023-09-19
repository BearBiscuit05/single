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


def evaluate(model, graph, dataloader):
    model.eval()
    ys = []
    y_hats = []
    for it, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
        with torch.no_grad():
            x = blocks[0].srcdata['feat']
            ys.append(blocks[-1].dstdata['label'].cpu().numpy())
            y_hats.append(model(blocks, x).argmax(1).cpu().numpy())
        predictions = np.concatenate(y_hats)
        labels = np.concatenate(ys)
    return sklearn.metrics.accuracy_score(labels, predictions)

def layerwise_infer(device, graph, nid, model, batch_size):
    model.eval()
    with torch.no_grad():
        pred = model.inference(graph, device, batch_size) # pred in buffer_device
        pred = pred[nid]
        label = graph.ndata['label'][nid].to(pred.device)
    return sklearn.metrics.accuracy_score(label.cpu().numpy(), pred.argmax(1).cpu().numpy())

def train(args, device, g, dataset, model,data=None ,basicLoop=0,loop=10):
    # create sampler & dataloader
    if data != None:
        train_idx,val_idx,test_idx = data 
    else:
        train_idx = dataset.train_idx.to(device)
        val_idx = dataset.val_idx.to(device)
        test_idx = dataset.test_idx.to(device)
    sampler = NeighborSampler(args.fanout,  # fanout for [layer-0, layer-1, layer-2]
                            prefetch_node_feats=['feat'],
                            prefetch_labels=['label'])
    use_uva = (args.mode == 'mixed')
    train_dataloader = DataLoader(g, train_idx, sampler, device=device,
                                  batch_size=1024, shuffle=True,
                                  drop_last=False, num_workers=0,
                                  use_uva=use_uva)

    val_dataloader = DataLoader(g, val_idx, sampler, device=device,
                                batch_size=1024, shuffle=True,
                                drop_last=False, num_workers=0,
                                use_uva=use_uva)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    
    for epoch in range(loop):
        model.train()
        total_loss = 0
        startTime = time.time()
        count = 0
        for it, (input_nodes, output_nodes, blocks) in enumerate(train_dataloader):
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
        acc = evaluate(model, g, val_dataloader)
        print("| Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} | Time {:.3f}s | Count {} |"
              .format(basicLoop+epoch, total_loss / (it+1), acc.item(), trainTime, count))
    
def load_reddit(self_loop=True):
    from dgl.data import RedditDataset
    data = RedditDataset(self_loop=self_loop,raw_dir='/home/bear/workspace/singleGNN/data/dataset/')
    g = data[0]
    g.ndata['feat'] = g.ndata.pop('feat')
    g.ndata['label'] = g.ndata.pop('label')
    train_idx = []
    val_idx = []
    test_idx = []
    for index in range(len(g.ndata['train_mask'])):
        if g.ndata['train_mask'][index] == 1:
            train_idx.append(index)
    for index in range(len(g.ndata['val_mask'])):
        if g.ndata['val_mask'][index] == 1:
            val_idx.append(index)
    for index in range(len(g.ndata['test_mask'])):
        if g.ndata['test_mask'][index] == 1:
            test_idx.append(index)
    return g, data,train_idx,val_idx,test_idx

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default='mixed', choices=['cpu', 'mixed', 'puregpu'],
                        help="Training mode. 'cpu' for CPU training, 'mixed' for CPU-GPU mixed training, "
                             "'puregpu' for pure-GPU training.")
    parser.add_argument('--fanout', type=ast.literal_eval, default=[15, 25], help='Fanout value')
    parser.add_argument('--layers', type=int, default=2, help='Number of layers')
    parser.add_argument('--dataset', type=str, default='ogb-products', help='Dataset name')
    parser.add_argument('--maxloop', type=int, default=200, help='max loop number')
    parser.add_argument('--model', type=str, default="SAGE", help='train model')
    args = parser.parse_args()
    if not torch.cuda.is_available():
        args.mode = 'cpu'
    print(f'Training in {args.mode} mode.')
    
    # load and preprocess dataset
    print('Loading data')
    if args.dataset == 'ogb-products':
        dataset = AsNodePredDataset(DglNodePropPredDataset('ogbn-products',root="/home/bear/workspace/singleGNN/data/dataset"))
        g = dataset[0]
        data = None
    elif args.dataset == 'Reddit':
        g, dataset,train_idx,val_idx,test_idx= load_reddit()
        data = (train_idx,val_idx,test_idx)
    elif args.dataset == 'ogb-papers100M':
        dataset = AsNodePredDataset(DglNodePropPredDataset('ogbn-papers100M',root="/home/bear/workspace/singleGNN/data/dataset"))
        g = dataset[0]
        data = None
    g = g.to('cuda' if args.mode == 'puregpu' else 'cpu')
    device = torch.device('cpu' if args.mode == 'cpu' else 'cuda')
    
    # create GraphSAGE model

    in_size = g.ndata['feat'].shape[1]
    out_size = dataset.num_classes
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
   

    # model training
    print('Training...')
    loopList = [0,10,20,30,50,100,150,200]
    for index in range(1,len(loopList)):
        if loopList[index] > args.maxloop:
            break
        _loop = loopList[index] - loopList[index - 1]
        train(args, device, g, dataset, model,data=data,basicLoop=loopList[index - 1],loop=_loop)
    # model = torch.load("save.pt")
    # model = model.to(device) 
    #model.load_state_dict(torch.load("model_param.pth"))
    # test the model
    
    
        print('Testing with after loop {}:...'.format(loopList[index]))
        if args.dataset == 'ogb-products':
            acc = layerwise_infer(device, g, dataset.test_idx, model, batch_size=4096)
        elif args.dataset == 'Reddit':
            acc = layerwise_infer(device, g, test_idx, model, batch_size=4096) 
        elif args.dataset == 'ogb-papers100M':
            model.eval()
            if args.layers == 2:
                sampler_test = NeighborSampler([100,100],  # fanout for [layer-0, layer-1, layer-2]
                                    prefetch_node_feats=['feat'],
                                    prefetch_labels=['label'])
            else:
                sampler_test = NeighborSampler([20,50,50],  # fanout for [layer-0, layer-1, layer-2]
                                    prefetch_node_feats=['feat'],
                                    prefetch_labels=['label'])
            test_dataloader = DataLoader(g, dataset.test_idx, sampler_test, device=device,
                                    batch_size=4096, shuffle=True,
                                    drop_last=False, num_workers=0,
                                    use_uva=True)
            acc = evaluate(model, g, test_dataloader)
        print("Test Accuracy {:.4f}".format(acc.item()))
        print("-"*20)