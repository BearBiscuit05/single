import os
import sys
import json
import time
import dgl
from torch.utils.data import Dataset, DataLoader
current_folder = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_folder+"/../"+"load")
from loader import CustomDataset

"比较dgl数据加载与自定义数据加载是否存在区别"

def datasetInit_test(dataset):
    "测试数据初始化加载部分是否有问题"

def subGLoading_test():
    "测试后续子图加载时候情况,包括halo部分"

def removSubG_test():
    "测试子图释放需要的时间"

def batch_test():
    "测试子图加载batch的正确性,与dgl的比较"

def nodeClassloading_test():
    "测试训练子图,测试子图,验证子图部分"

def nodeLabel_test():
    "测试标签准确性"

def haloSubG_test():
    "测试halo部分的准确性"

def sampleNeig():
    "测试采样部分的正确性"

def feat_test():
    "特征提取部分的正确性"

def changeMode_test():
    "模式切换测试"

def dglBlockTrans_test():
    "dgl转换后的测试验证"

def pygBlockTrans_test():
    "pyg转换后的测试验证部分"

def readGraph(rank,dataPath,datasetName):
    graph_dir = dataPath
    part_config = graph_dir + "/"+datasetName +'.json'
    print('loading partitions')
    subg, node_feat, _, gpb, _, node_type, _ = dgl.distributed.load_partition(part_config, rank)
    return subg, node_feat, node_type


def collate_fn(data):
    """
    data 输入结构介绍：
        [graph,feat]
    """
    return data[0]

if __name__ == '__main__':
    # dataset = CustomDataset("./../load/graphsage.json")
    
    # 常规数据加载
    # with open("./../load/graphsage.json", 'r') as f:
    #     config = json.load(f)
    #     batchsize = config['batchsize']
    #     epoch = config['epoch']
    # train_loader = DataLoader(dataset=dataset, batch_size=batchsize, collate_fn=collate_fn,pin_memory=True)
    # time.sleep(2)
    
    dataPath = "./../../data/raw-reddit_8"
    dataName = "reddit"
    savePath = "./../../data/reddit_8"
    subg, node_feat, node_type = readGraph(0,dataPath,dataName)
    print(subg.edges())
    # for index in range(epoch):
    #     count = 0
    #     for graph,feat,label,number in train_loader:
    #         #pass
    #         print("="*40)
    #         print("block:",graph)
    #         print("feat:",len(feat))
    #         print("label:",len(label))
    #         print("batch number :",number)
    #         exit()