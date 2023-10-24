from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix
import numpy as np 
import torch
import dgl
import time 
import copy


### config 
datasets = {
    "FR": {
        "GRAPHPATH": "/home/bear/workspace/single-gnn/data/raid/com_fr",
        "maxID": 65608366
    },
    "PA": {
        "GRAPHPATH": "/home/bear/workspace/single-gnn/data/raid/papers100M",
        "maxID": 111059956
    },
    "PD": {
        "GRAPHPATH": "/home/bear/workspace/single-gnn/data/raid/ogbn_products",
        "maxID": 2449029
    },
    "TW": {
        "GRAPHPATH": "/home/bear/workspace/single-gnn/data/raid/twitter",
        "maxID": 41652230
    },
    "UK": {
        "GRAPHPATH": "/home/bear/workspace/single-gnn/data/raid/uk-2006-05",
        "maxID": 77741046
    }
}


def acc_ana(tensor):
    num_ones = torch.sum(tensor == 1).item()  
    total_elements = tensor.numel()  
    percentage_ones = (num_ones / total_elements) * 100 
    print(f"only use by one train node : {percentage_ones:.2f}%")


    num_greater_than_1 = torch.sum(tensor > 1).item() 
    percentage_greater_than_1 = (num_greater_than_1 / total_elements) * 100
    print(f"use by multi train nodes : {percentage_greater_than_1:.2f}%")


## bfs 遍历获取基础子图



def analysisG(graph,maxID):
    src = torch.tensor(graph[::2])
    dst = torch.tensor(graph[1::2])
    ids = torch.arange(int(maxID*0.01),dtype=torch.int64)
    nodeTable = torch.zeros(maxID,dtype=torch.int32)
    nodeTable[ids] = 1

    batch_size = 1
    src_batches = torch.chunk(src, batch_size, dim=0)
    dst_batches = torch.chunk(dst, batch_size, dim=0)
    batch = [src_batches, dst_batches]

    repeats = 3
    acc = True
    start = time.time()
    for index in range(1,repeats+1):
        acc_tabel = torch.zeros_like(nodeTable,dtype=torch.int32)
        raw_nodeTabel = copy.deepcopy(nodeTable)
        print(f"before {index} BFS has {torch.nonzero(nodeTable).size(0)} nodes, "
            f"{torch.nonzero(nodeTable).size(0) * 1.0 / maxID * 100 :.2f}% of total nodes")
        for src_batch,dst_batch in zip(*batch):
            tmp_nodeTabel = copy.deepcopy(nodeTable)
            tmp_nodeTabel = tmp_nodeTabel.cuda()
            src_batch = src_batch.cuda()
            dst_batch = dst_batch.cuda()
            dgl.fastFindNeighbor(tmp_nodeTabel, src_batch, dst_batch, acc)
            tmp_nodeTabel = tmp_nodeTabel.cpu()
            acc_tabel = acc_tabel + tmp_nodeTabel - raw_nodeTabel
        print("end bfs...")
        acc_ana(acc_tabel)
        nodeTable = raw_nodeTabel + acc_tabel
        print(f"after {index} BFS has {torch.nonzero(nodeTable).size(0)} nodes, "
              f"{torch.nonzero(nodeTable).size(0) * 1.0 / maxID * 100 :.2f}% of total nodes")
        print('-'*10)
    print(f"all bfs cost {time.time()-start:.3f}s")

if __name__ == '__main__':
    selected_dataset = ["FR"] 
    for NAME in selected_dataset:
        dataset = datasets[NAME]
        GRAPHPATH = dataset["GRAPHPATH"]
        maxID = dataset["maxID"]
    graph = np.fromfile(GRAPHPATH+"/graph.bin",dtype=np.int32)
    analysisG(graph,maxID)