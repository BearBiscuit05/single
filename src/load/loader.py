import concurrent.futures
from torch.utils.data import Dataset, DataLoader
from queue import Queue
import numpy as np
import json
import time
import mmap
import dgl
import torch
from dgl.heterograph import DGLBlock
import random
import copy
import sys
import logging
import os
import gc
from tools import *
from memory_profiler import profile

curFilePath = os.path.abspath(__file__)
curDir = os.path.dirname(curFilePath)

# 禁用操作
#logging.disable(logging.CRITICAL)
logging.basicConfig(level=logging.INFO,filename=curDir+'/../../log/loader.log',filemode='w',
                    format='%(message)s',datefmt='%H:%M:%S')
                    #format='%(message)s')
logger = logging.getLogger(__name__)

"""
数据加载的逻辑:@profile
    1.生成训练随机序列
    2.预加载训练节点(所有的训练节点都被加载进入)
    2.预加载图集合(从初始开始就存入2个)
    3.不断生成采样子图
    4.当图采样完成后释放当前子图,加载下一个图
"""
class CustomDataset(Dataset):
    def __init__(self,confPath,pre_fetch=False):
        self.pre_fetch = pre_fetch
        
        self.cacheData = []  # 子图存储部分
        self.indptr = [] # dst / bound
        self.indices = [] # src / edgelist
        self.graphPipe = Queue()  # 采样存储管道
        self.sampleFlagQueue = Queue()
        self.executor = concurrent.futures.ThreadPoolExecutor(1) # 线程池      

        if self.pre_fetch == True:
            self.preFetchExecutor = concurrent.futures.ThreadPoolExecutor(1)  # 线程池
            self.preFetchFlagQueue = Queue()
            self.preFetchDataCache = Queue()
            self.pre_fetchGID = 0
            self.pregraph_src = 0
            self.pregraph_range = 0
            self.prefeat = 0
            self.preFetchPtr = 0  # 数据预取位置
            self.preFetchNUM = 0  # 数据预取数目

        #### config json 部分 ####
        self.dataPath = ''
        self.batchsize,self.cacheNUM,self.partNUM = 0,0,0
        self.maxEpoch,self.preRating,self.classes,self.epochInterval = 0,0,0,0
        self.featlen = 0
        self.fanout = []
        self.train_name,self.framework,self.mode,self.dataset = "","","",""
        self.readConfig(confPath)
        # ================

        #### 训练记录 ####
        self.trainSubGTrack = self.randomTrainList()    # 训练轨迹
        self.subGptr = -1  # 子图训练指针，记录当前训练的位置，在加载图时发生改变
        
        #### 节点类型加载 ####
        self.NodeLen = 0        # 用于记录数据集中节点的数目，默认为train节点个数
        self.trainNUM = 0       # 训练集总数目
        self.valNUM = 0
        self.testNUM = 0
        self.trainNodeDict,self.valNodeDict,self.testNodeDict = {},{},{}
        self.trainNodeNumbers,self.valNodeNumbers,self.testNodeNumbers = 0,0,0
        self.loadModeData(self.mode)

        #### 图结构信息 ####
        self.graphNodeNUM = 0  # 当前训练子图节点数目
        self.graphEdgeNUM = 0          # 当前训练子图边数目
        self.GID = 0            # 当前训练子图的ID
        self.subGtrainNodesNUM = 0      # 当前训练子图训练节点数目
        self.trainNodes = []            # 训练子图训练节点记录   
        self.nodeLabels = []            # 子图标签
        self.nextGID = 0                # 下一个训练子图
        self.trainptr = 0               # 当前训练集读取位置
        self.trainLoop = 0              # 当前子图可读取次数      
        #### mmap 特征部分 ####
        self.feats = []

        #### 数据预取 ####
        self.template_cache_graph= self.initCacheData()
        self.initNextGraphData()
        self.uniTable = torch.zeros(len(self.template_cache_graph[0])*2,dtype=torch.int32).to('cuda:0')

    def __len__(self):
        return self.NodeLen
    
    def __getitem__(self, index):
        if index % self.batchsize == 0:
            self.preGraphBatch()
            cacheData = self.graphPipe.get()
            return tuple(cacheData[:4])
        
        # if index % self.preRating == 0:
        #     self.sampleFlagQueue.put(self.executor.submit(self.preGraphBatch))
        
        # # 获取采样数据
        # if index % self.batchsize == 0:
        #     if self.graphPipe.qsize() > 0:
        #         self.sampleFlagQueue.get()
        #         cacheData = self.graphPipe.get()
        #         if self.train_name == "LP":
        #             return tuple(cacheData[:6])
        #         else:
        #             return tuple(cacheData[:4])
                
        #     else: #需要等待
        #         flag = self.sampleFlagQueue.get()
        #         flag.result()
        #         cacheData = self.graphPipe.get()
        #         if self.train_name == "LP":
        #             return tuple(cacheData[:6])
        #         else:
        #             return tuple(cacheData[:4])
        # return 0,0

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
        self.maxEpoch = config['maxEpoch']
        self.preRating = config['preRating']
        self.featlen = config['featlen']
        self.fanout = config['fanout']
        self.framework = config['framework']
        self.mode = config['mode']
        self.classes = config['classes']
        self.epochInterval = config['epochInterval']

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
        for loop in range(self.maxEpoch + 1):
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
                    for i in range(self.partNUM):
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
        for i in range(self.maxEpoch + 1): # 额外多增加一行
            random_array = np.random.choice(np.arange(0, self.partNUM), size=self.partNUM, replace=False)
            if len(epochList) == 0:
                epochList.append(random_array)
            else:
                # 已经存在列
                lastid = epochList[-1][-1]
                while(lastid == random_array[0]):
                    random_array = np.random.choice(np.arange(0, self.partNUM), size=self.partNUM, replace=False)
                epochList.append(random_array)

        # logger.info("train track:{}".format(epochList))    
        return epochList

########################## 加载/释放 图结构数据 ##########################
    def initNextGraphData(self):
        # 先拿到本次加载的内容，然后发送预取命令
        logger.info("----------initNextGraphData----------")
        start = time.time()
        
        self.subGptr += 1
        self.GID = self.trainSubGTrack[self.subGptr // self.partNUM][self.subGptr % self.partNUM]
        # 先判断是否为第一个，或者是已经存在发送过预取命令
        if self.subGptr == 0 or not self.pre_fetch:
            # 如果两个都没有，则全部从头加载
            self.loadingGraph(self.GID) # graph
            self.loadingMemFeat(self.GID) # feat
        elif self.pre_fetch:
            # 如果已经有预取命令，则获得预取数据，并补充完整剩下的内容
            prefetch = False
            if self.preFetchFlagQueue.qsize() > 0:
                prefetch = True
                taskFlag = self.preFetchFlagQueue.get()
                taskFlag.result()
                preCacheData = self.preFetchDataCache.get()
                self.loadingGraph(self.GID,preFetch=True,srcList=preCacheData[0],rangeList=preCacheData[1])
                self.loadingMemFeat(self.GID,preFetch=True,preFeat=preCacheData[2])

        self.loadingLabels(self.GID)
        self.trainNodes = self.trainNodeDict[self.GID]
        self.subGtrainNodesNUM = self.trainNodeNumbers[self.GID]   
        self.trainLoop = ((self.subGtrainNodesNUM - 1) // self.batchsize) + 1
        if self.pre_fetch:
            self.preFetchFlagQueue.put(self.preFetchExecutor.submit(self.preloadingGraphData))

    def loadingTrainID(self):
        # 加载子图所有训练集
        idDict = {}
        numberList = [0 for i in range(self.partNUM)]  
        for index in range(self.partNUM):
            filePath = self.dataPath + "/part" + str(index)   
            trainIDs = torch.tensor(np.fromfile(filePath+"/trainIds.bin",dtype=np.int64))
            idDict[index],_ = torch.sort(trainIDs)
            current_length = len(idDict[index])
            numberList[index] = current_length
            fill_length = self.batchsize - current_length % self.batchsize
            padding = torch.full((fill_length,), -1, dtype=idDict[index].dtype)
            idDict[index] = torch.cat((idDict[index], padding))
            logger.debug("subG:{} ,real train len:{}, padding number:{}".format(index,current_length,padding))
            self.trainNUM += idDict[index].shape[0]
        return idDict,numberList

    def preloadingGraphData(self):
        # 暂时只转换为numpy格式
        ptr = self.subGptr + 1
        rank = self.trainSubGTrack[ptr // self.partNUM][ptr % self.partNUM]
        filePath = self.dataPath + "/part" + str(rank)
        indices = np.fromfile(filePath + "/indices.bin", dtype=np.int32)
        indptr = np.fromfile(filePath + "/indptr.bin", dtype=np.int32)
        tmp_feat = np.fromfile(filePath + "/feat.bin", dtype=np.float32)
        self.preFetchDataCache.put([indices,indptr,tmp_feat])
        return 0

    def loadingGraph(self,subGID,preFetch=False,srcList=None,rangeList=None):
        """
        merge 表示此处需要融合两个图结构
        preFetch 表示下一需要融合的图已经被加载
        """
        if preFetch and (srcList is None or rangeList is None):
            raise ValueError("If preFetch is set to True, srcList and rangeList must not be None.")
        filePath = self.dataPath + "/part" + str(subGID)
        if preFetch == False:
            self.indices = torch.tensor(np.fromfile(filePath + "/indices.bin", dtype=np.int32), device=('cuda:0'))
            self.indptr = torch.tensor(np.fromfile(filePath + "/indptr.bin", dtype=np.int32), device=('cuda:0'))
        else:
            self.indices = torch.tensor(srcList, device=('cuda:0'))
            self.indptr = torch.tensor(rangeList, device=('cuda:0'))
        self.graphNodeNUM = int(len(self.indices) - 1 )
        self.graphEdgeNUM = len(self.indices)

    def loadingLabels(self,GID):
        filePath = self.dataPath + "/part" + str(GID)
        self.nodeLabels = torch.tensor(np.fromfile(filePath+"/labels.bin", dtype=np.int64))

    def loadingMemFeat(self, rank, preFetch=False,preFeat=None):
        self.feats = []
        torch.cuda.empty_cache()
        gc.collect()
        if preFetch and preFeat is None:
            raise ValueError("If preFetch is set to True, preFeat must not be None.")
        filePath = self.dataPath + "/part" + str(rank)
        if preFetch == False:
            preFeat = np.fromfile(filePath + "/feat.bin", dtype=np.float32)
        self.feats = torch.from_numpy(preFeat).reshape(-1, self.featlen).to("cuda:0")

########################## 采样图结构 ##########################
    def sampleNeig(self,sampleIDs,cacheGraph): 
        layer = len(self.fanout)
        for l, number in enumerate(self.fanout):
            number -= 1
            if l != 0:     
                last_lens = len(cacheGraph[layer-l][0])      
                lastids = cacheGraph[layer - l][0]
            else:
                last_lens = len(sampleIDs)
                lastids = sampleIDs
            cacheGraph[layer-l-1][0][0:last_lens] = lastids
            cacheGraph[layer-l-1][1][0:last_lens] = lastids
            for index in range(len(lastids)):
                ids = cacheGraph[layer-l-1][0][index]
                if ids == -1:
                    continue
                try:
                    NeigList = self.cacheData[0][self.cacheData[1][ids*2]+1:self.cacheData[1][ids*2+1]]
                except:
                    logger.error("error: srcLen:{},rangelen:{},ids:{}".format(len(self.cacheData[0]),len(self.cacheData[1]),ids))
                    exit(-1)

                if len(NeigList) < number:
                    sampled_values = NeigList
                else:
                    sampled_values = np.random.choice(NeigList,number)
                
                offset = last_lens + (index * number)
                fillsize = len(sampled_values)
                cacheGraph[layer-l-1][0][offset:offset+fillsize] = sampled_values # src
                cacheGraph[layer-l-1][1][offset:offset+fillsize] = [ids] * fillsize # dst
        for info in cacheGraph:
            info[0] = torch.tensor(info[0])
            info[1] = torch.tensor(info[1])

    def sampleNeigGPU_NC(self,sampleIDs,cacheGraph,batchlen):     
        logger.info("----------[sampleNeigGPU_NC]----------")
        sampleIDs = sampleIDs.to(torch.int32).to('cuda:0')
        ptr = 0
        mapping_ptr = [ptr]
        
        sampleStart = time.time()
        for l, fan_num in enumerate(self.fanout):
            if l == 0:
                seed_num = batchlen
            else:
                seed_num = len(sampleIDs)
            out_src = cacheGraph[0][ptr:ptr+seed_num*fan_num]
            out_dst = cacheGraph[1][ptr:ptr+seed_num*fan_num]
            
            NUM = dgl.sampling.sample_with_edge(self.indptr,self.indices,
                sampleIDs,seed_num,fan_num,out_src,out_dst)
            sampleIDs = cacheGraph[0][ptr:ptr+NUM.item()]
            ptr=ptr+NUM.item()
            mapping_ptr.append(ptr)

        logger.info("Sample Neighbor Time {:.5f}s".format(time.time()-sampleStart))
        mappingTime = time.time()        
        cacheGraph[0] = cacheGraph[0][:mapping_ptr[-1]]
        cacheGraph[1] = cacheGraph[1][:mapping_ptr[-1]]

        unique = copy.deepcopy(self.uniTable)

        logger.info("construct remapping data Time {:.5f}s".format(time.time()-mappingTime))
        t = time.time()  
        cacheGraph[0],cacheGraph[1],unique = dgl.remappingNode(cacheGraph[0],cacheGraph[1],unique)
        logger.info("cuda remapping func Time {:.5f}s".format(time.time()-t))

        transTime = time.time()
        if self.framework == "dgl":
            layer = len(mapping_ptr) - 1
            blocks = []
            save_num = 0
            for index in range(1,layer+1):
                src = cacheGraph[0][:mapping_ptr[-index]]
                dst = cacheGraph[1][:mapping_ptr[-index]]
                data = (src,dst)
                if index == 1:
                    save_num,_ = torch.max(dst,dim=0)
                    save_num += 1
                    block = self.create_dgl_block(data,len(unique),save_num)
                elif index == layer:
                    tmp_num = save_num
                    save_num,_ = torch.max(dst,dim=0)
                    save_num += 1
                    block = self.create_dgl_block(data,tmp_num,save_num)
                else:
                    tmp_num = save_num
                    save_num,_ = torch.max(dst,dim=0)
                    save_num += 1
                    block = self.create_dgl_block(data,tmp_num,save_num)
                blocks.append(block)
        elif self.framework == "pyg":
            src = cacheGraph[0][:mapping_ptr[-1]].to(torch.int64)
            dst = cacheGraph[1][:mapping_ptr[-1]].to(torch.int64)
            blocks = torch.stack((src, dst), dim=0)
        logger.info("trans Time {:.5f}s".format(time.time()-transTime))
        
        
        logger.info("==>sampleNeigGPU_NC() func time {:.5f}s".format(time.time()-sampleStart))
        logger.info("-"*30)
        return blocks,unique
    
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
            tmp = tmp * (fan + 1)

        cacheGraph[0] = torch.cat(cacheGraph[0],dim=0)
        cacheGraph[1] = torch.cat(cacheGraph[1],dim=0)
        return cacheGraph

    def preGraphBatch(self):
        preBatchTime = time.time()
        if self.graphPipe.qsize() >= self.cacheNUM:
            return 0
        if self.trainptr == self.trainLoop:
            logger.debug("触发cache reload ,ptr:{}".format(self.trainptr))
            self.trainptr = 0           
            self.initNextGraphData()
        cacheTime = time.time()
        cacheGraph = copy.deepcopy(self.template_cache_graph)
        # TODO: 
        sampleIDs = -1 * torch.ones(self.batchsize,dtype=torch.int64)
        logger.info("construct copy graph and label cost {:.5f}s".format(time.time()-cacheTime))
        
        createDataTime = time.time()
        batchlen = 0
        if self.trainptr < self.trainLoop - 1:
            # 完整batch
            sampleIDs = self.trainNodes[self.trainptr*self.batchsize:(self.trainptr+1)*self.batchsize]
            batchlen = self.batchsize
            cacheLabel = self.nodeLabels[sampleIDs]
        else:
            offset = self.trainptr*self.batchsize
            sampleIDs = self.trainNodes[offset:self.subGtrainNodesNUM]
            batchlen = self.subGtrainNodesNUM - offset
            cacheLabel = self.nodeLabels[sampleIDs]
        logger.info("prepare sample data Time cost {:.5f}s".format(time.time()-createDataTime))    

        ##
        sampleTime = time.time()
        if self.train_name == "LP":
            negNUM = 1
            uniqueSeed,raw_edges,src_cat,dst_cat = self.getNegNode(sampleIDs,batchlen,negNUM=negNUM)
            batchlen = len(uniqueSeed)
            blocks,uniqueList = self.sampleNeigGPU_LP(uniqueSeed,raw_edges,cacheGraph,batchlen)
        else:
            blocks,uniqueList = self.sampleNeigGPU_NC(sampleIDs,cacheGraph,batchlen)
        logger.info("sample subG all cost {:.5f}s".format(time.time()-sampleTime))
        ##
     
        cacheFeat = self.featMerge(uniqueList)
        
        if self.train_name == "LP":
            cacheData = [blocks,cacheFeat,cacheLabel,src_cat,dst_cat,batchlen]
        else:
            cacheData = [blocks,cacheFeat,cacheLabel,batchlen]
        self.graphPipe.put(cacheData)
        
        self.trainptr += 1
        logger.info("-"*30)
        logger.info("preGraphBatch() cost {:.5f}s".format(time.time()-preBatchTime))
        logger.info("="*30)
        logger.info("\t")
        return 0

########################## 特征提取 ##########################
    def featMerge(self,uniqueList):
        featTime = time.time()           
        test = self.feats[uniqueList.to(torch.int64).to(self.feats.device)]     
        logger.info("subG feat merge cost {:.5f}s".format(time.time()-featTime))
        return test

########################## 数据调整 ##########################    
    def loadModeData(self,mode):
        logger.info("loading mode:'{}' data".format(mode))
        self.trainNodeDict,self.trainNodeNumbers = self.loadingTrainID() # 训练节点字典，训练节点数目
        self.NodeLen = self.trainNUM
 
    def create_dgl_block(self, data, num_src_nodes, num_dst_nodes):
        row, col = data
        gidx = dgl.heterograph_index.create_unitgraph_from_coo(2, num_src_nodes, num_dst_nodes, row, col, 'coo')
        g = DGLBlock(gidx, (['_N'], ['_N']), ['_E'])
        return g

def collate_fn(data):
    return data[0]



if __name__ == "__main__":
    dataset = CustomDataset(curDir+"/../../config/train_config.json",pre_fetch=True)
    with open(curDir+"/../../config/train_config.json", 'r') as f:
        config = json.load(f)
        batchsize = config['batchsize']
        epoch = config['maxEpoch']
    train_loader = DataLoader(dataset=dataset, batch_size=batchsize,collate_fn=collate_fn)
    count = 0
    for index in range(2):
        start = time.time()
        loopTime = time.time()
        for graph,feat,label,number in train_loader:
            # feat.to("cuda:0")
            # src_cat,dst_cat,
            # print(graph)
            # print(src_cat)
            # print(dst_cat)
            # exit()
            count = count + 1
            if count % 20 == 0:
                print("loop time:{:.5f}".format(time.time()-loopTime))
            loopTime = time.time()