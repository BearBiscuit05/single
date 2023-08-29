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
from memory_profiler import profile



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
        print(self.trainSubGTrack)
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
        self.template_cache_graph,self.template_cache_label = self.initCacheData()
        self.loadingGraph(merge=False)
        self.loadingMemFeat(self.trainSubGTrack[self.subGptr//self.partNUM][self.subGptr%self.partNUM])
        self.initNextGraphData()
        
    def __len__(self):  
        return self.NodeLen
    
    def __getitem__(self, index):
        # 批数据预取 缓存1个
        
        if index % self.preRating == 0:
            # 调用预取函数
            self.sampleFlagQueue.put(self.executor.submit(self.preGraphBatch))
        
        # 获取采样数据
        if index % self.batchsize == 0:
            if self.graphPipe.qsize() > 0:
                self.sampleFlagQueue.get()
                cacheData = self.graphPipe.get()
                return cacheData[0],cacheData[1],cacheData[2],cacheData[3]
                
            else: #需要等待
                flag = self.sampleFlagQueue.get()
                data = flag.result()
                cacheData = self.graphPipe.get()
                return cacheData[0],cacheData[1],cacheData[2],cacheData[3]
        return 0,0,0,0

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
        formatted_data = json.dumps(config, indent=4)

    def custom_sort(self):
        idMap={}
        for i in range(self.partNUM):
            folder_path = self.dataPath+"/part"+str(i)
            idMap[i] = []
            for filename in os.listdir(folder_path):
                if filename.startswith("halo") and filename.endswith(".bin"):
                    try:
                        x = int(filename[len("halo"):-len(".bin")])
                        idMap[i].append(x)
                    except:
                        continue

        sorted_numbers = []
        lastid = 0
        for loop in range(self.epoch + 1):
            used_numbers = set()
            tmp = []
            for idx in range(0,self.partNUM):
                if idx == 0:
                    num = lastid
                else:
                    num = tmp[-1]
                candidates = idMap[num]
                available_candidates = [int(candidate) for candidate in candidates if int(candidate) not in used_numbers]                
                if available_candidates:
                    chosen_num = random.choice(available_candidates)
                    tmp.append(chosen_num)
                    used_numbers.add(chosen_num)
                else:
                    for i in range(partNUM):
                        if i not in used_numbers:
                            available_candidates.append(i)
                    chosen_num = random.choice(available_candidates)
                    tmp.append(chosen_num)
                    used_numbers.add(chosen_num)
            sorted_numbers.append(tmp)
            lastid = tmp[-1]
        print(sorted_numbers)
        return sorted_numbers

    def randomTrainList(self): 
        #epochList = self.custom_sort()
        epochList = []
        for i in range(10): # 额外多增加一行
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
        if self.subGptr > 0:
            self.moveGraph()
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

        self.loadingGraph()
        self.nextGID = self.trainSubGTrack[self.subGptr//self.partNUM][self.subGptr%self.partNUM]
        halostart = time.time()
        haloend = time.time()
        print("loading.....=====================>")
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
            #logger.debug("subG:{} ,real train len:{}, padding number:{}".format(index,current_length,padding))
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
        # print("self.subGptr:",self.subGptr)
        # print(self.subGptr//self.partNUM,"  -  ",self.subGptr%self.partNUM)
        subGID = self.trainSubGTrack[self.subGptr//self.partNUM][self.subGptr%self.partNUM]
        filePath = self.dataPath + "/part" + str(subGID)
        srcdata = np.fromfile(filePath+"/srcList.bin", dtype=np.int32)
        srcdata = torch.tensor(srcdata,device=('cuda:%d'%self.cudaDevice))#.to(device=('cuda:%d'%self.cudaDevice))
        rangedata = np.fromfile(filePath+"/range.bin", dtype=np.int32)
        rangedata = torch.tensor(rangedata,device=('cuda:%d'%self.cudaDevice))#.to(device=('cuda:%d'%self.cudaDevice))
        
        #print(type(srcdata))
        #print(type(rangedata))
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

    #@profile(precision=4, stream=open('./info.log','w+'))
    def moveGraph(self):
        self.cacheData[0] = self.cacheData[0][self.graphEdgeNUM:]   # 边
        self.cacheData[1] = self.cacheData[1][self.graphNodeNUM*2:]   # 范围
        self.cacheData[0] = self.cacheData[0] - self.graphNodeNUM   # 边 nodeID
        self.cacheData[1] = self.cacheData[1] - self.graphEdgeNUM
        self.feats = self.feats[self.graphNodeNUM:]  

########################## 采样图结构 ##########################
    def getNegNode(self,sampleIDs,batchlen,negNUM=1):
        sampleIDs = sampleIDs.to(torch.int32).to('cuda:0')
        out_src = torch.zeros(batchlen).to(torch.int32).to('cuda:0')
        out_dst = torch.zeros(batchlen).to(torch.int32).to('cuda:0')
        seed_num = batchlen
        fan_num = 1
        out_num = torch.Tensor([0]).to(torch.int64).to('cuda:0')
        # print("sample 1111")
        # print(sampleIDs)
        signn.torch_sample_hop(
                self.cacheData[0][:self.graphEdgeNUM],self.cacheData[1][:self.graphNodeNUM*2],
                sampleIDs,seed_num,fan_num,
                out_src,out_dst,out_num)
        # print("sample 1122")
        out_src = out_src[:out_num.item()]
        out_dst = out_dst[:out_num.item()]
        raw_src = copy.deepcopy(out_src)
        raw_dst = copy.deepcopy(out_dst)
        # print(raw_src.shape + raw_src.shape)

        # print()
        # exit()
        neg_dst = torch.randint(low=0, high=self.graphNodeNUM, size=raw_src.shape).to(torch.int32).to("cuda:0")
        
        all_tensor = torch.cat([raw_src,raw_dst,raw_src,neg_dst])
        raw_edges = torch.cat([raw_src,raw_dst])
        src_cat = torch.cat([raw_src,raw_src])
        dst_cat = torch.cat([raw_dst,neg_dst])
        raw_src = copy.deepcopy(out_src)
        raw_dst = copy.deepcopy(out_dst)
        edgeNUM = len(src_cat)     
        uniqueNUM = torch.Tensor([0]).to(torch.int64).to('cuda:0')
        unique = torch.zeros(len(all_tensor),dtype=torch.int32).to('cuda:0')

        # t_min,_ = torch.max(raw_src,dim=0)
        # t1_max,_ = torch.max(raw_dst,dim=0)
        # t2_max,_ = torch.max(neg_dst,dim=0)
        # print("raw_src max :",t_min,"  raw_dst max :",t1_max,"  neg_dst max :",t2_max)
        # print("all_tensor:",all_tensor," shape :",all_tensor.shape)
        signn.torch_graph_mapping(all_tensor,src_cat,dst_cat,src_cat,dst_cat,unique,edgeNUM,uniqueNUM)
        # if uniqueNUM.item() > 
        # print("uniqueNUM.item():",uniqueNUM.item())
        # print("unique: ",unique[:uniqueNUM.item()])
        return unique[:uniqueNUM.item()],raw_edges,src_cat,dst_cat
    
    #@profile(precision=4, stream=open('./initCacheData.log','w+'))
    def initCacheData(self):
        if self.train_name == "NC":
            number = self.batchsize
        else:
            number = self.batchsize * 3
        tmp = number
        cacheGraph = [[],[]]
        for layer, fan in enumerate(self.fanout):
            dst = torch.full((tmp * fan,), -1, dtype=torch.int32).to("cuda:0")  # 使用PyTorch张量，指定dtype
            src = torch.full((tmp * fan,), -1, dtype=torch.int32).to("cuda:0")  # 使用PyTorch张量，指定dtype
            cacheGraph[0].append(src)
            cacheGraph[1].append(dst)
            tmp = tmp * fan
        cacheLabel = torch.zeros(self.batchsize)
        cacheGraph[0] = torch.cat(cacheGraph[0],dim=0)
        cacheGraph[1] = torch.cat(cacheGraph[1],dim=0)
        return cacheGraph, cacheLabel

    #@profile(precision=4, stream=open('./info.log','w+'))
    # 无影响
    def preGraphBatch(self):
        self.trainptr = 0           
        self.initNextGraphData()

        # cacheTime = time.time()
        # cacheGraph = copy.deepcopy(self.template_cache_graph)
        # cacheLabel = copy.deepcopy(self.template_cache_label)
        # sampleIDs = -1 * torch.ones(self.batchsize,dtype=torch.int64)

        # createDataTime = time.time()
        # batchlen = 0
        # if self.trainptr < self.trainLoop - 1:
        #     # 完整batch
        #     sampleIDs = self.trainNodes[self.trainptr*self.batchsize:(self.trainptr+1)*self.batchsize]
        #     batchlen = self.batchsize
        #     cacheLabel = self.nodeLabels[sampleIDs]
        # else:
        #     offset = self.trainptr*self.batchsize
        #     sampleIDs[:self.subGtrainNodesNUM - offset] = self.trainNodes[offset:self.subGtrainNodesNUM]
        #     batchlen = self.subGtrainNodesNUM - offset
        #     cacheLabel = self.nodeLabels[sampleIDs[0:self.subGtrainNodesNUM - offset]]
        
        cacheData = [[],[],[],[]]
        self.graphPipe.put(cacheData)
        
        self.trainptr += 1
        return 0



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
            self.feats = torch.from_numpy(tmp_feat).reshape(-1,self.featlen)
        else:
            tmp_feat = torch.from_numpy(tmp_feat).reshape(-1,self.featlen)
            self.feats = torch.cat([self.feats,tmp_feat])
    
    #@profile(precision=4, stream=open('./info.log','w+'))
    def featMerge(self,uniqueList):    
        featTime = time.time() 
        test = self.feats[uniqueList.to(torch.int64).to('cpu')]     
        return test
    
    def loadModeData(self,mode):
        #logger.info("loading mode:'{}' data".format(mode))
        if "train" == mode:
            self.trainNodeDict,self.trainNodeNumbers = self.loadingTrainID() # 训练节点字典，训练节点数目
            self.NodeLen = self.trainNUM
        elif "val" == mode:
            self.valNodeDict,self.valNodeNumbers = self.loadingValID() # 训练节点字典，训练节点数目
            self.NodeLen = self.valNUM
        elif "test" == mode:
            self.testNodeDict,self.testNodeNumbers = self.loadingTestID() # 训练节点字典，训练节点数目
            self.NodeLen = self.testNUM

def collate_fn(data):
    """
    data 输入结构介绍：
        [graph,feat]
    """
    return data[0]



if __name__ == "__main__":
    dataset = CustomDataset("../../config/dgl_papers_graphsage.json")
    with open("../../config/dgl_papers_graphsage.json", 'r') as f:
        config = json.load(f)
        batchsize = config['batchsize']
        epoch = config['epoch']
    train_loader = DataLoader(dataset=dataset, batch_size=batchsize,collate_fn=collate_fn)#pin_memory=True)
    count = 0
    for index in range(2):
        start = time.time()
        loopTime = time.time()
        for graph,feat,label,number in train_loader:
            count = count + 1
            if count % 20 == 0:
                print("loop time:{:.5f}".format(time.time()-loopTime))
            loopTime = time.time()