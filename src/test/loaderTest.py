import os
import sys
import json
import time
import dgl
import dgl.nn as dglnn
from dgl.data import AsNodePredDataset
from ogb.nodeproppred import DglNodePropPredDataset

from torch.utils.data import Dataset, DataLoader
current_folder = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_folder+"/../"+"load")
sys.path.append("/home/bear/workspace/singleGNN/data/")
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

def load_wholegraph():
    dataset = AsNodePredDataset(DglNodePropPredDataset('ogbn-products'))
    g = dataset[0]
    g = g.to('cuda:0')

def haloSubG_test(partConfigJson,cacheData,graphNodeNum,trainingGID,nextGID,cudaDeviceIndex):
    "测试halo部分的准确性"
    with open(partConfigJson,'r') as f:
        SUBGconf = json.load(f)
        # 使用读取的数据
    boundRange = SUBGconf['node_map']['_N']
    
    edges = cacheData[0]
    nodeNum = len(cacheData[1])//2
    edgeNum = len(cacheData[0])
    srcTensor = torch.zeros(edgeNum,dtype=torch.int32,device=('cuda:%d'%cudaDeviceIndex))
    dstTensor = torch.zeros(edgeNum,dtype=torch.int32,device=('cuda:%d'%cudaDeviceIndex))
    cnt = 0
    # 当前子图
    for dstid in range(self.graphNodeNUM):
        boundl,boundr = cacheData[1][2*dstid],cacheData[1][2*dstid+1]
        srcids = cacheData[0][boundl:boundr]
        # (srcid->dstid)是两个子图拼接以后的边，节点id是转换后的，注意需要转换成全局id
        for index,srcid in enumerate(srcids):
            local_srcid = srcid
            local_srcid = dstid
            global_srcid = local_srcid + boundRange[trainingGID][0]
            global_dstid = local_dstid + boundRange[trainingGID][0]
            srcTensor[cnt] = global_srcid
            dstTensor[cnt] = global_dstid
            cnt = cnt + 1
    
    # 下一个子图
    for dstid in range(self.graphNodeNUM,nodeNum):
        boundl,boundr = cacheData[1][2*dstid],cacheData[1][2*dstid+1]
        srcids = cacheData[0][boundl:boundr]
        # (srcid->dstid)是两个子图拼接以后的边，节点id是转换后的，注意需要转换成全局id
        for index,srcid in enumerate(srcids):
            local_srcid = srcid - self.graphNodeNum
            local_dstid = dstid - self.graphNodeNum
            global_srcid = local_srcid + boundRange[self.nextGID][0]
            global_dstid = local_dstid + boundRange[self.nextGID][0]
            srcTensor[cnt] = global_srcid
            dstTensor[cnt] = global_dstid
            cnt = cnt + 1
    
    # dgl加载全图到GPU，用has_edges_between检测边的存在性
    dataset = AsNodePredDataset(DglNodePropPredDataset('ogbn-products'))
    g = dataset[0]
    g = g.to('cuda:0')
    comp = cuda_subg.has_edges_between(srcTensor,dstTensor)
    if comp.all() == False:
        print('loadingHalo Test failed')


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
    
    # dataPath = "./../../data/raw-reddit_8"
    # dataName = "reddit"
    # savePath = "./../../data/reddit_8"
    # subg, node_feat, node_type = readGraph(0,dataPath,dataName)
    # print(subg.edges())
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

    load_wholegraph()