import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
import dgl
import dgl.nn as dglnn
#from torch.utils.data import Dataset, DataLoader
import ast
import random
import copy
import tqdm
import argparse
import sklearn.metrics
import numpy as np
import time
import sys
import os
import json
from dgl.data import AsNodePredDataset
from ogb.nodeproppred import DglNodePropPredDataset
from dgl.dataloading import NeighborSampler, MultiLayerFullNeighborSampler

current_folder = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_folder+"/../../"+"load")
#from loader_dgl import CustomDataset
from loader import CustomDataset

class SAGE(nn.Module):
    def __init__(self, in_size, hid_size, out_size,layer_NUM=2):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(in_size, hid_size, 'mean'))
        for _ in range(layer_NUM - 2):
            self.layers.append(dglnn.SAGEConv(hid_size, hid_size, 'mean'))
        self.layers.append(dglnn.SAGEConv(hid_size, out_size, 'mean'))
        self.dropout = nn.Dropout(0.5)
        self.hid_size = hid_size
        self.out_size = out_size

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):         
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        
        return h

    def inference(self, g, device, batch_size):
        """Conduct layer-wise inference to get all the node embeddings."""
        feat = g.ndata['feat']
        sampler = MultiLayerFullNeighborSampler(1, prefetch_node_feats=['feat'])
        dataloader = dgl.dataloading.DataLoader(
                g, torch.arange(g.num_nodes()).to(g.device), sampler, device=device,
                batch_size=batch_size, shuffle=False, drop_last=False,
                num_workers=0)
        buffer_device = torch.device('cpu')
        pin_memory = (buffer_device != device)

        for l, layer in enumerate(self.layers):
            y = torch.empty(
                g.num_nodes(), self.hid_size if l != len(self.layers) - 1 else self.out_size,
                device=buffer_device, pin_memory=pin_memory)
            feat = feat.to(device)
            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                x = feat[input_nodes]
                h = layer(blocks[0], x) # len(blocks) = 1
                if l != len(self.layers) - 1:
                    h = F.relu(h)
                    h = self.dropout(h)
                # by design, our output nodes are contiguous
                y[output_nodes[0]:output_nodes[-1]+1] = h.to(buffer_device)
            feat = y
        return y
    
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
    #label = label.squeeze()
    #pred = pred.squeeze()
    return sklearn.metrics.accuracy_score(label.cpu().numpy(), pred.argmax(1).cpu().numpy())

def collate_fn(data):
    """
    data 输入结构介绍：
        [graph,feat]
    """
    return data[0]

def train(arg_dataset,device,dataset,model):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1024, collate_fn=collate_fn)#,pin_memory=True)
    # 修改此处，epoch数必须同步修改json文件里的epoch数
    epochTime = [0]
    testEpoch = [5,30,50,100,200]
    for epoch in range(1,dataset.epoch+1):
        startTime = time.time()
        total_loss = 0
        model.train()
        for it,(graph,feat,label,number) in enumerate(train_loader):
            # print(graph)
            # exit()
            feat = feat.to('cuda:0')
            tmp = copy.deepcopy(graph)
            tmp = [block.to('cuda:0') for block in tmp]
            y_hat = model(tmp, feat)
            try:
                loss = F.cross_entropy(y_hat[:number], label[:number].to(torch.int64).to('cuda:0'))
            except:
                print("error info : y_hat :{} , label :{} ,number :{}".format(y_hat.shape,label.shape,number))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        eptime = time.time() - startTime
        totTime = epochTime[epoch-1] + eptime
        epochTime.append(totTime)
        print("Epoch {:03d} | Loss {:.4f} | Time {:.6f}s".format(epoch, total_loss / (it+1), eptime))
        if epoch in testEpoch:
            test(arg_dataset)
    print("Average Training Time of {:d} Epoches:{:.6f}".format(dataset.epoch,epochTime[dataset.epoch]/dataset.epoch))
    print("Total   Training Time of {:d} Epoches:{:.6f}".format(dataset.epoch,epochTime[dataset.epoch]))

def load_reddit(self_loop=True):
    from dgl.data import RedditDataset
    data = RedditDataset(self_loop=self_loop,raw_dir='../../../data/dataset/')
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

def test(arg_dataset):
    if arg_dataset == 'ogb-products':
        dataset = AsNodePredDataset(DglNodePropPredDataset('ogbn-products'))
        g = dataset[0]
        begTime = time.time()
        acc = layerwise_infer(device, g, dataset.test_idx, model, batch_size=4096)
        endTime = time.time()
    elif arg_dataset == 'Reddit':
        g,dataset,train_idx,val_idx,test_idx= load_reddit()
        begTime = time.time()
        acc = layerwise_infer(device, g, test_idx, model, batch_size=4096)
        endTime = time.time()
    print("Test Accuracy {:.4f},Test Time {:.6f}".format(acc.item(),endTime-begTime))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default='mixed', choices=['cpu', 'mixed', 'puregpu'],
                        help="Training mode. 'cpu' for CPU training, 'mixed' for CPU-GPU mixed training, "
                             "'puregpu' for pure-GPU training.")
    parser.add_argument('--json_path', type=str, default='.', help='Dataset name')
    args = parser.parse_args()

    if not torch.cuda.is_available():
        args.mode = 'cpu'
    
    data = None
    with open(args.json_path, 'r') as json_file:
        data = json.load(json_file)

    print(f'Training in {args.mode} mode.')
    print('Loading data')
    if data["dataset"] == "products_4":
        arg_dataset = 'ogb-products'
    elif data["dataset"] == "reddit_8":
        arg_dataset = 'Reddit'
    else:
        raise ValueError(f"Unsupported dataset")
    
    arg_fanout = data["fanout"]
    arg_layers = len(arg_fanout)

    device = torch.device('cpu' if args.mode == 'cpu' else 'cuda:0')
    model = SAGE(data['featlen'], 256, data['classes'],arg_layers).to('cuda:0')  # 请确保 SAGE 模型的参数正确
    dataset = CustomDataset(args.json_path)  # 使用 args.json_path 作为 JSON 文件路径
    print('Training...')
    train(arg_dataset,device,dataset,model)

    # 指定要保存的文件路径
    # save_path = 'model_parameters.pth'

    # 保存模型参数
    # torch.save(model.state_dict(), save_path)
