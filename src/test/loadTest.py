import os
import sys
import json
import time
from torch.utils.data import Dataset, DataLoader
current_folder = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_folder+"/../"+"load")
from loaderTmp import CustomDataset

def collate_fn(data):
    """
    data 输入结构介绍：
        [graph,feat]
    """
    return data[0]

if __name__ == '__main__':
    dataset = CustomDataset("./../load/config.json")
    with open("./../load/config.json", 'r') as f:
        config = json.load(f)
        batchsize = config['batchsize']
        epoch = config['epoch']
    train_loader = DataLoader(dataset=dataset, batch_size=batchsize, collate_fn=collate_fn,pin_memory=True)
    time.sleep(2)
    
    for index in range(epoch):
        count = 0
        for graph,feat,label,number in train_loader:
            #pass
            print("="*40)
            print("block:",graph)
            print("feat:",len(feat))
            print("label:",len(label))
            print("batch number :",number)
            exit()