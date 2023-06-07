import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import dgl
import struct
import time

"""
tensor -> dgl.block

merge to dataloader , we get two tensor

tensor1: edgeLists  -> [src,dst]
tensor2: feat       -> same dim with src
"""

class MyDataset(Dataset):
    def __init__(self,trainIDs):
        self.data = [i for i in range(10000)]
        self.bound = [i*100 for i in range(100)]
        self.trainIDs = trainIDs
        self.preSample = []

    def __getitem__(self, index):
        #用来进行提前采样
        trainID = self.trainIDs[index]
        src = self.data[self.bound[trainID]:self.bound[trainID+1]]
        return src
 
    def __len__(self):
        return len(self.trainIDs)


def collate_fn(data):
    return data

if __name__ == "__main__":
    trainIDs = [1,2,3,4]
    data = MyDataset(trainIDs) 
    start = time.time()
    train_loader = DataLoader(dataset=data, batch_size=4, collate_fn=collate_fn,pin_memory=True)
    for i in train_loader:
        print(i)
    print(time.time()-start)
