import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
import dgl
import dgl.nn as dglnn
from dgl.data import AsNodePredDataset
from torch.utils.data import Dataset, DataLoader
from ogb.nodeproppred import DglNodePropPredDataset
import tqdm
import argparse
import sklearn.metrics
import numpy as np
import time
import time
import sys
import os
import copy
from dgl.data import AsNodePredDataset
from dgl.dataloading import NeighborSampler, MultiLayerFullNeighborSampler
current_folder = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_folder+"/../../"+"load")
#from loader_dgl import CustomDataset
from loader import CustomDataset
class GAT(nn.Module):
    def __init__(self,in_size, hid_size, out_size, heads):
        super().__init__()
        self.layers = nn.ModuleList()
        # two-layer GAT
        self.heads = heads
        self.layers.append(dglnn.GATConv(in_size, hid_size, heads[0], feat_drop=0.6, attn_drop=0.6, activation=F.elu,allow_zero_in_degree=True))
        self.layers.append(dglnn.GATConv(hid_size*heads[0], out_size, heads[1], feat_drop=0.6, attn_drop=0.6, activation=None,allow_zero_in_degree=True))
        self.hid_size = hid_size
        self.out_size = out_size

    def forward(self, g, inputs):
        h = inputs
        for i, layer in enumerate(self.layers):
            h = layer(g[i], h)
            if i == 1:  # last layer 
                h = h.mean(1)
            else:       # other layer(s)
                h = h.flatten(1)
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
                g.num_nodes(), self.hid_size*self.heads[0] if l != len(self.layers) - 1 else self.out_size,
                device=buffer_device, pin_memory=pin_memory)
            feat = feat.to(device)
            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                x = feat[input_nodes]
                h = layer(blocks[0], x) # len(blocks) = 1
                if l == 1:  # last layer 
                    h = h.mean(1)
                else:       # other layer(s)
                    h = h.flatten(1)
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
            tmp = copy.deepcopy(graph)
            tmp = [block.to('cuda:0') for block in tmp]
            y_hat = model(tmp, feat.to('cuda:0'))
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
    parser.add_argument("--mode", default='mixed', choices=['cpu', 'mixed', 'puregpu'],
                        help="Training mode. 'cpu' for CPU training, 'mixed' for CPU-GPU mixed training, "
                             "'puregpu' for pure-GPU training.")
    parser.add_argument('--fanout', type=ast.literal_eval, default=[25, 10], help='Fanout value')
    parser.add_argument('--layers', type=int, default=2, help='Number of layers')
    parser.add_argument('--dataset', type=str, default='Reddit', help='Dataset name')
    parser.add_argument('--json_path', type=str, default='.', help='Dataset name')
    args = parser.parse_args()

    if not torch.cuda.is_available():
        args.mode = 'cpu'
    print(f'Training in {args.mode} mode.')
    
    print('Loading data')
    data = None
    with open(args.json_path, 'r') as json_file:
        data = json.load(json_file)
    device = torch.device('cpu' if args.mode == 'cpu' else 'cuda')

    # create GraphSAGE model
    # in_size = g.ndata['feat'].shape[1]
    # out_size = dataset.num_classes
    print('Loading data')

    device = torch.device('cpu' if args.mode == 'cpu' else 'cuda')
    model = GCN(data['featlen'], 256, data['classes'] ,args.layers,F.relu,0.5).to('cuda:0')

    model = GAT(data['featlen'], 256, data['classes'], heads=[8,1]).to('cuda:0')

    # model training
    print('Training...')
    dataset = CustomDataset(args.json_path)  # 使用 args.json_path 作为 JSON 文件路径
    train(args, device, dataset, model)
    # train(args, device, g, dataset, model,data=data)

    if args.dataset == 'ogb-products':
        dataset = AsNodePredDataset(DglNodePropPredDataset('ogbn-products'))
        g = dataset[0]
        data = None
    elif args.dataset == 'Reddit':
        g, dataset,train_idx,val_idx,test_idx= load_reddit()
        data = (train_idx,val_idx,test_idx)

    if args.dataset == 'ogb-products':
        acc = layerwise_infer(device, g, dataset.test_idx, model, batch_size=4096)
    elif args.dataset == 'Reddit':
        acc = layerwise_infer(device, g, test_idx, model, batch_size=4096) 
    print("Test Accuracy {:.4f}".format(acc.item()))