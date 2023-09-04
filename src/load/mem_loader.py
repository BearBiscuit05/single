import concurrent.futures
from torch.utils.data import Dataset, DataLoader
from queue import Queue
import numpy as np
import json
import time
import mmap
import dgl
import torch
import torch
from dgl.heterograph import DGLBlock
import random
import copy
import sys
import logging
import signn
import os
import gc

class CustomDataset(Dataset):
    #@profile(precision=4, stream=open('./__init__.log','w+'))
    def __init__(self,confPath):
        #### 采样资源 ####
        self.cacheData = []     # 子图存储部分
        self.graphPipe = Queue()    # 采样存储管道
        self.sampleFlagQueue = Queue()
        self.executor = concurrent.futures.ThreadPoolExecutor(1) # 线程池
        
        #### config json 部分 ####
        self.dataPath = ''
        self.batchsize = 0
        self.cacheNUM = 0
        self.partNUM = 0
        self.epoch = 0
        self.preRating = 0
        self.featlen = 0
        self.idbound = []
        self.fanout = []
        self.train_name = ""
        self.framework = ""
        self.mode = ""
        self.dataset = ""
        self.classes = 0
        self.readConfig(confPath)
        # ================

        #### 训练记录 ####
        self.trainSubGTrack = self.randomTrainList()    # 训练轨迹
        self.subGptr = -1                               # 子图训练指针，记录当前训练的位置，在加载图时发生改变
        
        #### 节点类型加载 ####
        self.NodeLen = 0        # 用于记录数据集中节点的数目，默认为train节点个数
        self.trainNUM = 0       # 训练集总数目
        self.valNUM = 0
        self.testNUM = 0
        self.trainNodeDict,self.valNodeDict,self.testNodeDict = {},{},{}
        self.trainNodeNumbers,self.valNodeNumbers,self.testNodeNumbers = 0,0,0
        self.loadModeData(self.mode)

        #### 图结构信息 ####
        self.graphNodeNUM = 0           # 当前训练子图节点数目
        self.graphEdgeNUM = 0           # 当前训练子图边数目
        self.trainingGID = 0            # 当前训练子图的ID
        self.subGtrainNodesNUM = 0      # 当前训练子图训练节点数目
        self.trainNodes = []            # 训练子图训练节点记录   
        self.nodeLabels = []            # 子图标签
        self.nextGID = 0                # 下一个训练子图
        self.trainptr = 0               # 当前训练集读取位置
        self.trainLoop = 0              # 当前子图可读取次数
        
        #### mmap 特征部分 ####
        self.readfile = []              # 包含两个句柄/可能有三个句柄
        self.mmapfile = []  
        self.feats = []
        
        #### 规定用哪张卡单独跑 ####
        self.cudaDevice = 0


        #### 数据预取 ####
        # self.template_cache_graph,self.template_cache_label = self.initCacheData()
        self.loadingGraph(merge=False)
        self.loadingMemFeat(self.trainSubGTrack[self.subGptr//self.partNUM][self.subGptr%self.partNUM])
        print("next init graph")
        self.initNextGraphData()
        #self.sampleFlagQueue.put(self.executor.submit(self.preGraphBatch)) #发送采样命令
        
    def __len__(self):  
        return self.NodeLen
    
    def __getitem__(self, index):
        if index % self.batchsize == 0:
            return [],[],[],[]
        return 0,0

########################## 初始化训练数据 ##########################
    def readConfig(self,confPath):
        with open(confPath, 'r') as f:
            config = json.load(f)
        self.train_name = config['train_name']
        self.dataPath = config['datasetpath']+"/"+config['dataset']
        self.dataset = config['dataset']
        self.batchsize = config['batchsize']
        self.cacheNUM = config['cacheNUM']
        self.partNUM = config['partNUM']
        self.epoch = config['epoch']
        self.preRating = config['preRating']
        self.featlen = config['featlen']
        self.fanout = config['fanout']
        self.idbound = config['idbound']
        self.framework = config['framework']
        self.mode = config['mode']
        self.classes = config['classes']
        #print(formatted_data)

    def randomTrainList(self): 
        epochList = []
        for i in range(self.epoch + 1): # 额外多增加一行
            random_array = np.random.choice(np.arange(0, self.partNUM), size=self.partNUM, replace=False)
            if len(epochList) == 0:
                epochList.append(random_array)
            else:
                # 已经存在列
                lastid = epochList[-1][-1]
                while(lastid == random_array[0]):
                    random_array = np.random.choice(np.arange(0, self.partNUM), size=self.partNUM, replace=False)
                epochList.append(random_array)

        return epochList

########################## 加载/释放 图结构数据 ##########################
    #@profile(precision=4, stream=open('./info.log','w+'))
    def initNextGraphData(self):
        start = time.time()
        # 查看是否需要释放
        if self.subGptr > 0:
            self.moveGraph()
        # 对于将要计算的子图(已经加载)，修改相关信息
        self.trainingGID = self.trainSubGTrack[self.subGptr//self.partNUM][self.subGptr%self.partNUM]
        self.graphNodeNUM = int(len(self.cacheData[1]) / 2 )# 获取当前节点数目
        self.graphEdgeNUM = len(self.cacheData[0])
        self.nodeLabels = self.loadingLabels(self.trainingGID)  
        # 节点设置部分
        if "train" == self.mode:
            self.trainNodes = self.trainNodeDict[self.trainingGID]
            self.subGtrainNodesNUM = self.trainNodeNumbers[self.trainingGID]   
        elif "val" == self.mode:
            self.trainNodes = self.valNodeDict[self.trainingGID]
            self.subGtrainNodesNUM = self.valNodeNumbers[self.trainingGID]  
        elif "test" == self.mode:
            self.trainNodes = self.testNodeDict[self.trainingGID]
            self.subGtrainNodesNUM = self.testNodeNumbers[self.trainingGID]
        self.trainLoop = ((self.subGtrainNodesNUM - 1) // self.batchsize) + 1

        # 对于辅助计算的子图，进行加载，以及加载融合边
        self.loadingGraph()
        self.nextGID = self.trainSubGTrack[self.subGptr//self.partNUM][self.subGptr%self.partNUM]
        halostart = time.time()
        haloend = time.time()
        self.loadingMemFeat(self.nextGID)

    def loadingTrainID(self):
        # 加载子图所有训练集
        idDict = {}
        numberList = [0 for i in range(self.partNUM)]  
        for index in range(self.partNUM):
            filePath = self.dataPath + "/part" + str(index)   
            trainIDs = torch.load(filePath+"/trainID.bin")
            # trainIDs = trainIDs.to(torch.uint8).nonzero().squeeze()[:TESTNODE]
            trainIDs = trainIDs.to(torch.uint8).nonzero().squeeze()
            idDict[index],_ = torch.sort(trainIDs)
            current_length = len(idDict[index])
            numberList[index] = current_length
            fill_length = self.batchsize - current_length % self.batchsize
            padding = torch.full((fill_length,), -1, dtype=idDict[index].dtype)
            idDict[index] = torch.cat((idDict[index], padding))
            self.trainNUM += idDict[index].shape[0]
        return idDict,numberList

    def loadingValID(self):
        # 加载子图所有训练集
        idDict = {}
        numberList = [0 for i in range(self.partNUM)]  
        for index in range(self.partNUM):
            filePath = self.dataPath + "/part" + str(index)   
            ValIDs = torch.load(filePath+"/valID.bin")
            ValIDs = ValIDs.to(torch.uint8).nonzero().squeeze()
            idDict[index],_ = torch.sort(ValIDs)
            current_length = len(idDict[index])
            numberList[index] = current_length
            fill_length = self.batchsize - current_length % self.batchsize
            padding = torch.full((fill_length,), -1, dtype=idDict[index].dtype)
            idDict[index] = torch.cat((idDict[index], padding))
            self.valNUM += idDict[index].shape[0]
        return idDict,numberList

    def loadingTestID(self):
        # 加载子图所有训练集
        idDict = {}
        numberList = [0 for i in range(self.partNUM)]  
        for index in range(self.partNUM):
            filePath = self.dataPath + "/part" + str(index)   
            TestID = torch.load(filePath+"/testID.bin")
            TestID = TestID.to(torch.uint8).nonzero().squeeze()
            idDict[index],_ = torch.sort(TestID)
            current_length = len(idDict[index])
            numberList[index] = current_length
            fill_length = self.batchsize - current_length % self.batchsize
            padding = torch.full((fill_length,), -1, dtype=idDict[index].dtype)
            idDict[index] = torch.cat((idDict[index], padding))
            self.testNUM += idDict[index].shape[0]
        return idDict,numberList

    #@profile(precision=4, stream=open('./info.log','w+'))
    def loadingGraph(self,merge=True):
        # 加载下一个等待训练的图
        self.subGptr += 1
        subGID = self.trainSubGTrack[self.subGptr//self.partNUM][self.subGptr%self.partNUM]
        filePath = self.dataPath + "/part" + str(subGID)
        srcdata = np.fromfile(filePath+"/srcList.bin", dtype=np.int32)
        srcdata = torch.tensor(srcdata,device='cpu')#.to(device=('cuda:%d'%self.cudaDevice))
        rangedata = np.fromfile(filePath+"/range.bin", dtype=np.int32)
        rangedata = torch.tensor(rangedata,device='cpu')#.to(device=('cuda:%d'%self.cudaDevice))

        if merge :
            srcdata = srcdata + self.graphNodeNUM
            rangedata = rangedata + self.graphEdgeNUM
            self.cacheData[0] = torch.cat([self.cacheData[0],srcdata])
            self.cacheData[1] = torch.cat([self.cacheData[1],rangedata])
        else:
            # 第一次加载
            self.cacheData.append(srcdata)
            self.cacheData.append(rangedata)
        
    def loadingLabels(self,rank):
        filePath = self.dataPath + "/part" + str(rank)
        if self.dataset == "papers100M_64":
            labels = torch.from_numpy(np.fromfile(filePath+"/label.bin", dtype=np.int32)).to(torch.int64)
        else:
            labels = torch.from_numpy(np.fromfile(filePath+"/label.bin", dtype=np.int32)).to(torch.int64)
        return labels

    def moveGraph(self):
        print("testing")
        self.cacheData[0] = self.cacheData[0][self.graphEdgeNUM:]   # 边
        self.cacheData[1] = self.cacheData[1][self.graphNodeNUM*2:]   # 范围
        self.cacheData[0] = self.cacheData[0] - self.graphNodeNUM   # 边 nodeID
        self.cacheData[1] = self.cacheData[1] - self.graphEdgeNUM
        self.feats = self.feats[self.graphNodeNUM:]   
        gc.collect()


    def initCacheData(self):
        if self.train_name == "NC":
            number = self.batchsize
        else:
            number = self.batchsize * 3
        tmp = number
        cacheGraph = [[],[]]
        for layer, fan in enumerate(self.fanout):
            dst = torch.full((tmp * fan,), -1, dtype=torch.int32).to("cpu")  # 使用PyTorch张量，指定dtype
            src = torch.full((tmp * fan,), -1, dtype=torch.int32).to("cpu")  # 使用PyTorch张量，指定dtype
            cacheGraph[0].append(src)
            cacheGraph[1].append(dst)
            tmp = tmp * (fan + 1)


        cacheLabel = torch.zeros(self.batchsize)
        cacheGraph[0] = torch.cat(cacheGraph[0],dim=0)
        cacheGraph[1] = torch.cat(cacheGraph[1],dim=0)
        return cacheGraph, cacheLabel

########################## 特征提取 ##########################
    def loadingFeatFileHead(self):
        for index in range(self.partNUM):
            filePath = self.dataPath + "/part" + str(index)
            file = open(filePath+"/feat.bin", "r+b")
            self.readfile.append(file)
            self.mmapfile.append(mmap.mmap(self.readfile[-1].fileno(), 0, access=mmap.ACCESS_DEFAULT))

    def closeMMapFileHead(self):
        for file in self.mmapfile:
            file.close()
        for file in self.readfile:
            file.close()

    #@profile(precision=4, stream=open('./info.log','w+'))
    def loadingMemFeat(self,rank):
        filePath = self.dataPath + "/part" + str(rank)
        tmp_feat = np.fromfile(filePath+"/feat.bin", dtype=np.float32)
        if self.feats == []:
            self.feats = torch.from_numpy(tmp_feat).reshape(-1,self.featlen)#.to("cpu")
        else:
            tmp_feat = torch.from_numpy(tmp_feat).reshape(-1,self.featlen)#.to("cpu")
            #self.feats = torch.cat([self.feats,tmp_feat])

    def featMerge(self,uniqueList):    
        
        featTime = time.time() 
        test = self.feats[uniqueList.to(torch.int64).to('cpu')]     
        return test
    
    def loadModeData(self,mode):
        if "train" == mode:
            self.trainNodeDict,self.trainNodeNumbers = self.loadingTrainID() # 训练节点字典，训练节点数目
            self.NodeLen = self.trainNUM
        elif "val" == mode:
            self.valNodeDict,self.valNodeNumbers = self.loadingValID() # 训练节点字典，训练节点数目
            self.NodeLen = self.valNUM
        elif "test" == mode:
            self.testNodeDict,self.testNodeNumbers = self.loadingTestID() # 训练节点字典，训练节点数目
            self.NodeLen = self.testNUM

   
    def create_dgl_block(self, data, num_src_nodes, num_dst_nodes):
        row, col = data
        gidx = dgl.heterograph_index.create_unitgraph_from_coo(2, num_src_nodes, num_dst_nodes, row, col, 'coo')
        g = DGLBlock(gidx, (['_N'], ['_N']), ['_E'])
        return g

def collate_fn(data):
    """
    data 输入结构介绍：
        [graph,feat]
    """
    return data[0]



if __name__ == "__main__":
    dataset = CustomDataset("/home/bear/workspace/singleGNN/config/dgl_papers_graphsage.json")
    with open("/home/bear/workspace/singleGNN/config/dgl_papers_graphsage.json", 'r') as f:
        config = json.load(f)
        batchsize = config['batchsize']
        epoch = config['epoch']
    train_loader = DataLoader(dataset=dataset, batch_size=batchsize,collate_fn=collate_fn)#pin_memory=True)
