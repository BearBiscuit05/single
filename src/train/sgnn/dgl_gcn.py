import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
import dgl
import dgl.nn as dglnn
from torch.utils.data import Dataset, DataLoader
import random
from dgl.nn.pytorch import GraphConv
import copy
import tqdm
import argparse
import sklearn.metrics
import numpy as np
import time
import sys
import os
from dgl.data import AsNodePredDataset
from ogb.nodeproppred import DglNodePropPredDataset
from dgl.dataloading import NeighborSampler, MultiLayerFullNeighborSampler

current_folder = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_folder+"/../../"+"load")
from loader import CustomDataset

class GCN(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        self.hid_size = n_hidden
        self.out_size = n_classes
        self.layers.append(GraphConv(in_feats, n_hidden, activation=activation, allow_zero_in_degree=True))
        for _ in range(1, n_layers - 1):
            self.layers.append(
                GraphConv(n_hidden, n_hidden, activation=activation, allow_zero_in_degree=True))
        self.layers.append(
            GraphConv(n_hidden, n_classes, allow_zero_in_degree=True))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, blocks, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(blocks[i], h)
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
    return sklearn.metrics.accuracy_score(label.cpu().numpy(), pred.argmax(1).cpu().numpy())

def train(args, device, dataset, model):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    train_loader = DataLoader(dataset=dataset, batch_size=1024, collate_fn=collate_fn)
    for epoch in range(dataset.epoch):
        start = time.time()
        total_loss = 0
        model.train()
        
        for it,(graph,feat,label,number) in enumerate(train_loader):
            #print(graph)
            tmp = copy.deepcopy(graph)
            tmp = [block.to('cuda:0') for block in tmp]
            y_hat = model(graph, feat.to('cuda:0'))
            #print("y_hat len:{},label len:{},number:{}".format(len(y_hat),len(label),number))
            loss = F.cross_entropy(y_hat[:number], label[:number].to(torch.int64).to('cuda:0'))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print("Epoch {:05d} | Loss {:.4f} | Time {:.3f}s"
              .format(epoch, total_loss / (it+1), time.time()-start))
        # acc = evaluate(model, g, val_dataloader)
        # print("Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} "
        #       .format(epoch, total_loss / (it+1), acc.item()))
        # print("time :",time.time()-start)



def collate_fn(data):
    """
    data 输入结构介绍：
        [graph,feat]
    """
    return data[0]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default='puregpu', choices=['cpu', 'mixed', 'puregpu'],
                        help="Training mode. 'cpu' for CPU training, 'mixed' for CPU-GPU mixed training, "
                             "'puregpu' for pure-GPU training.")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        args.mode = 'cpu'
    print(f'Training in {args.mode} mode.')
    
    # load and preprocess dataset
    print('Loading data')
    # dataset = AsNodePredDataset(DglNodePropPredDataset('ogbn-products'))
    # g = dataset[0]
    # g = g.to('cuda' if args.mode == 'puregpu' else 'cpu')
    device = torch.device('cpu' if args.mode == 'cpu' else 'cuda')

    # create GraphSAGE model
    # in_size = g.ndata['feat'].shape[1]
    # out_size = dataset.num_classes
    model = GCN(100, 256, 47 ,2,F.relu,0.5).to('cuda:0')

    # model training
    print('Training...')
    dataset = CustomDataset("./../../load/graphsage.json")
    train(args, device, dataset, model)
    # train(args, device, g, dataset, model,data=data)

    # # test the model
    test_dataset = AsNodePredDataset(DglNodePropPredDataset('ogbn-products'))
    print('Testing...')
    g = test_dataset[0]
    g = g.to('cpu')
    #acc = layerwise_infer(device, g, test_idx, model, batch_size=4096)
    acc = layerwise_infer(device, g, test_dataset.test_idx, model, batch_size=4096)
    print("Test Accuracy {:.4f}".format(acc.item()))