import torch
import dgl
from ogb.nodeproppred import DglNodePropPredDataset
from dgl.data import AsNodePredDataset
from loader import CustomDataset
import json
import time
from torch.utils.data import Dataset, DataLoader

def neriBlockCompare(dgl_graph,sgnn_block,nowGraphId,nextGraphId,batchsize):
    dataset = AsNodePredDataset(DglNodePropPredDataset('ogbn-products'))
    g = dataset[0]

def collate_fn(data):
    return data[0]

def turnToGID(ids,nowid,nextid,bound):
    nodeNUM = bound[nowid][1] - bound[nowid][0]
    nowGraphBeginId = bound[nowid][1]
    nextGraphBeginId = bound[nextid][1]
    processed_vector = np.where(ids >= nodeNUM, lambda x: x-nodeNUM+nextGraphBeginId, lambda x: x + nowGraphBeginId)

def useLoading(g):
    bound = []
    with open("./ogb-product.json", 'r') as f:
        config = json.load(f)
        bound = config['node_map']["_N"]
    
    dataset = CustomDataset("./graphsage.json")
    with open("./graphsage.json", 'r') as f:
        config = json.load(f)
        batchsize = config['batchsize']
        epoch = config['epoch']
    train_loader = DataLoader(dataset=dataset, batch_size=batchsize,collate_fn=collate_fn)
    count = 0
    for index in range(1):
        start = time.time()
        loopTime = time.time()
        for graph,feat,label,number,nowId,nextId in train_loader:      
            src = graph[0][0]
            dst = graph[0][1]
            src = turnToGID(src,nowId,nextId,bound)
            dst = turnToGID(dst,nowId,nextId,bound)
            print(src)
            print(dst)
            loopTime = time.time()
        print("compute time:{}".format(time.time()-start))


if __name__ == '__main__':
    dataset = AsNodePredDataset(DglNodePropPredDataset('ogbn-products'))
    g = dataset[0]
    useLoading(g)