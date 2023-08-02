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
import sample_hop
import os

logging.basicConfig(level=logging.INFO,filename='./log/loader.log',filemode='w',
                    format='%(asctime)s-%(levelname)s-%(message)s',datefmt='%H:%M:%S')
                    #format='%(message)s')
logger = logging.getLogger(__name__)

# TESTNODE = 3000
#变量控制原则 : 谁用谁负责 
"""
数据加载的逻辑:
    1.生成训练随机序列
    2.预加载训练节点(所有的训练节点都被加载进入)
    2.预加载图集合(从初始开始就存入2个)
    3.不断生成采样子图
    4.当图采样完成后释放当前子图,加载下一个图
"""
class CustomDataset(Dataset):
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
        self.framework = ""
        self.mode = ""
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
        self.loadingFeatFileHead()      # 读取特征文件
        
        #### 测试，临时规定用哪张卡单独跑 ####
        self.cudaDevice = 0

        #### dgl.block ####
        if self.framework == "dgl":
            self.templateBlock = self.genBlockTemplate()
        elif self.framework == "pyg":
        #### pyg.batch ####
            self.templateBlock = self.genPYGBatchTemplate()

        #### 数据预取 ####
        self.loadingGraph(merge=False)
        self.loadingMemFeat(self.trainSubGTrack[self.subGptr//self.partNUM][self.subGptr%self.partNUM])
        self.initNextGraphData()
        self.sampleFlagQueue.put(self.executor.submit(self.preGraphBatch)) #发送采样命令
        
        

    def __len__(self):  
        return self.NodeLen
    
    def __getitem__(self, index):
        # 批数据预取 缓存1个
        if index % self.preRating == 0:
            # 调用预取函数
            self.sampleFlagQueue.put(self.executor.submit(self.preGraphBatch))
        
        # 获取采样数据
        if index % self.batchsize == 0:
            # 调用实际数据
            if self.graphPipe.qsize() > 0:
                self.sampleFlagQueue.get()
                cacheData = self.graphPipe.get()
                return cacheData[0],cacheData[1],cacheData[2],cacheData[3]
            else: #需要等待
                flag = self.sampleFlagQueue.get()
                data = flag.result()
                cacheData = self.graphPipe.get()
                return cacheData[0],cacheData[1],cacheData[2],cacheData[3]
        return 0,0

########################## 初始化训练数据 ##########################
    def readConfig(self,confPath):
        with open(confPath, 'r') as f:
            config = json.load(f)
        self.dataPath = config['datasetpath']+"/"+config['dataset']
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
        formatted_data = json.dumps(config, indent=4)
        print(formatted_data)

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

        logger.info("train track:{}".format(epochList))    
        return epochList

########################## 加载/释放 图结构数据 ##########################
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
        self.loadingHalo()
        haloend = time.time()
        logger.info("loadingHalo time: %g"%(haloend-halostart))
        self.loadingMemFeat(self.nextGID)
        logger.info("当前加载图为:{},下一个图:{},图训练集规模:{},图节点数目:{},图边数目:{},加载耗时:{}s"\
                        .format(self.trainingGID,self.nextGID,self.subGtrainNodesNUM,\
                        self.graphNodeNUM,self.graphEdgeNUM,time.time()-start))

    def loadingTrainID(self):
        # 加载子图所有训练集
        idDict = {}
        numberList = [0 for i in range(self.partNUM)]  
        for index in range(self.partNUM):
            filePath = self.dataPath + "/part" + str(index)   
            trainIDs = torch.load(filePath+"/trainID.bin")
            trainIDs = trainIDs.to(torch.uint8).nonzero().squeeze()
            _,idDict[index] = torch.sort(trainIDs)
            idDict[index] = idDict[index]
            current_length = len(idDict[index])
            numberList[index] = current_length
            fill_length = self.batchsize - current_length % self.batchsize
            padding = torch.full((fill_length,), -1, dtype=idDict[index].dtype)
            idDict[index] = torch.cat((idDict[index], padding))
            logger.debug("subG:{} ,real train len:{}, padding number:{}".format(index,current_length,padding))
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
            _,idDict[index] = torch.sort(ValIDs)
            idDict[index] = idDict[index]
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
            _,idDict[index] = torch.sort(TestID)
            idDict[index] = idDict[index]
            current_length = len(idDict[index])
            numberList[index] = current_length
            fill_length = self.batchsize - current_length % self.batchsize
            padding = torch.full((fill_length,), -1, dtype=idDict[index].dtype)
            idDict[index] = torch.cat((idDict[index], padding))
            self.testNUM += idDict[index].shape[0]
        return idDict,numberList

    def loadingGraph(self,merge=True):
        # 加载下一个等待训练的图
        self.subGptr += 1
        subGID = self.trainSubGTrack[self.subGptr//self.partNUM][self.subGptr%self.partNUM]
        filePath = self.dataPath + "/part" + str(subGID)
        srcdata = np.fromfile(filePath+"/srcList.bin", dtype=np.int32)
        srcdata = torch.tensor(srcdata).to(device=('cuda:%d'%self.cudaDevice))
        rangedata = np.fromfile(filePath+"/range.bin", dtype=np.int32)
        rangedata = torch.tensor(rangedata).to(device=('cuda:%d'%self.cudaDevice))
        
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
        return torch.from_numpy(np.fromfile(filePath+"/label.bin", dtype=np.int32)).to(torch.int64)

    def moveGraph(self):
        logger.debug("move last graph {},and now graph {}".format(self.trainingGID,self.nextGID))
        logger.debug("befor move srclist len:{}".format(len(self.cacheData[0])))
        logger.debug("befor move range len:{}".format(len(self.cacheData[1])))
        self.cacheData[0] = self.cacheData[0][self.graphEdgeNUM:]   # 边
        self.cacheData[1] = self.cacheData[1][self.graphNodeNUM*2:]   # 范围
        self.cacheData[0] = self.cacheData[0] - self.graphNodeNUM   # 边 nodeID
        self.cacheData[1] = self.cacheData[1] - self.graphEdgeNUM
        self.feats = self.feats[self.graphNodeNUM:]
        logger.debug("after move srclist len:{}".format(len(self.cacheData[0])))
        logger.debug("after move range len:{}".format(len(self.cacheData[1])))     

    def loadingHalo(self):
        # 要先加载下一个子图，然后再加载halo( 当前<->下一个 )
        filePath = self.dataPath + "/part" + str(self.trainingGID) 
        edges = np.fromfile(filePath+"/halo"+str(self.nextGID)+".bin", dtype=np.int32)
        edges = torch.tensor(edges).to(torch.int32).to(device=('cuda:%d'%self.cudaDevice))#.contiguous()
        bound = np.fromfile(filePath+"/halo"+str(self.nextGID)+"_bound.bin", dtype=np.int32)
        bound = torch.tensor(bound).to(torch.int32).to(device=('cuda:%d'%self.cudaDevice))#.contiguous()
        self.cacheData[0] = self.cacheData[0]#.contiguous()
        self.cacheData[1] = self.cacheData[1]#.contiguous()
        # lastid = torch.tensor(-1).to(device=('cuda:%d'%self.cudaDevice))
        # startidx = torch.tensor(-1).to(device=('cuda:%d'%self.cudaDevice))
        # endidx = torch.tensor(-1).to(device=('cuda:%d'%self.cudaDevice))
        # nextidx = torch.tensor(-1).to(device=('cuda:%d'%self.cudaDevice))
        # #srcList = self.cacheData[0]
        # #bound = self.cacheData[1]
        # for index in range(int(len(edges) / 2)):
        #     src = edges[index*2] 
        #     dst = edges[index*2 + 1]
        #     # print(src.device)
        #     # print(dst.device)
        #     if dst != lastid:
        #         startidx = self.cacheData[1][dst*2]
        #         endidx = self.cacheData[1][dst*2+1]
        #         try:
        #             next = self.cacheData[1][dst*2+2]
        #         except:
        #             next = torch.tensor(self.graphEdgeNUM).to(device=('cuda:%d'%self.cudaDevice))
        #         lastid = dst
        #         if endidx < next:
        #             self.cacheData[0][endidx] = src
        #             endidx += 1
        #     else:
        #         if endidx < next:
        #             self.cacheData[0][endidx] = src
        #             endidx += 1
        sample_hop.torch_launch_loading_halo_new(self.cacheData[0],self.cacheData[1],edges,bound,len(self.cacheData[0]),len(self.cacheData[1]),len(edges),len(bound),self.graphEdgeNUM,3)
        #sample_hop.torch_launch_loading_halo0(self.cacheData[0],self.cacheData[1],edges,len(self.cacheData[0]),len(self.cacheData[1]),len(edges),self.graphEdgeNUM,3)

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

    def sampleNeigGPU(self,sampleIDs,cacheGraph,cudaDeviceIndex):
        # 设定GPU设备
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1' #调试用

        if torch.cuda.is_available() == False:
            print('No GPU Device')
            return self.sampleNeig(sampleIDs,cacheGraph)
        elif cudaDeviceIndex < 0 or cudaDeviceIndex >= torch.cuda.device_count():
            #print('Wrong GPU Index Argument %d, Select default 0'%cudaDeviceIndex)
            cudaDeviceIndex = 0
        deviceName = 'cuda:%d' % cudaDeviceIndex

        layer = len(self.fanout)

        srcdata   = self.cacheData[0]#.to(torch.int32).to(device=deviceName)
        rangedata = self.cacheData[1]#.to(torch.int32).to(device=deviceName)

        batchlen = 0
        if self.trainptr < self.trainLoop - 1:
            # 完整batch
            batchlen = self.batchsize
        else:
            # 最后一个batch
            offset = self.trainptr*self.batchsize
            batchlen = self.subGtrainNodesNUM - offset
        
        trainlist = torch.Tensor(sampleIDs[0:batchlen]).to(torch.int32).to(device=deviceName)

        for info in cacheGraph:
            info[0] = torch.tensor(info[0]).to(torch.int32).to(device=deviceName)#.contiguous()
            info[1] = torch.tensor(info[1]).to(torch.int32).to(device=deviceName)#.contiguous()
        
        #sample_start = time.time()
        if layer == 3:
            sample_hop.torch_launch_sample_3hop_new(
                cacheGraph[0][0],cacheGraph[0][1],cacheGraph[1][0],cacheGraph[1][1],cacheGraph[2][0],cacheGraph[2][1],
                srcdata, rangedata, trainlist,
                self.fanout[0],self.fanout[1],self.fanout[2],batchlen,cudaDeviceIndex)
        elif layer == 2:
                sample_hop.torch_launch_sample_2hop_new(
                cacheGraph[0][0],cacheGraph[0][1],cacheGraph[1][0],cacheGraph[1][1],
                srcdata, rangedata, trainlist,
                self.fanout[0],self.fanout[1],batchlen,cudaDeviceIndex)
        elif layer == 1:
            sample_hop.torch_launch_sample_1hop_new(
                cacheGraph[0][0],cacheGraph[0][1],
                srcdata, rangedata, trainlist,
                self.fanout[0],batchlen,cudaDeviceIndex)
        #sample_end = time.time()
        #print("sample%dHop time in Python:%g s"%(layer,sample_end-sample_start))

    def initCacheData(self):
        number = self.batchsize
        tmp = self.batchsize
        cacheGraph = []
        for layer, fan in enumerate(self.fanout):
            dst = [-1] * tmp * fan
            src = [-1] * tmp * fan
            cacheGraph.insert(0,[src,dst])
            tmp = tmp * fan
        cacheLabel = torch.zeros(self.batchsize)
        return cacheGraph,cacheLabel

    def preGraphBatch(self):
        # 如果当前管道已经被充满，则不采样，该函数直接返回
        if self.graphPipe.qsize() >= self.cacheNUM:
            return 0

        if self.trainptr == self.trainLoop:
            # 当前cache已经失效，则需要reload新图
            logger.debug("触发cache reload ,ptr:{}".format(self.trainptr))
            self.trainptr = 0           
            self.initNextGraphData()
                   
        cacheGraph,cacheLabel = self.initCacheData()
        sampleIDs = [-1] * self.batchsize
        batchlen = 0
        if self.trainptr < self.trainLoop - 1:
            # 完整batch
            sampleIDs = self.trainNodes[self.trainptr*self.batchsize:(self.trainptr+1)*self.batchsize]
            batchlen = self.batchsize
            cacheLabel = self.nodeLabels[sampleIDs]
        else:
            # 最后一个batch
            logger.debug("last batch...")
            offset = self.trainptr*self.batchsize
            logger.debug("train loop:{} , offset:{} ,subGtrainNodesNUM:{}".format(self.trainLoop,offset,self.subGtrainNodesNUM))
            sampleIDs[:self.subGtrainNodesNUM - offset] = self.trainNodes[offset:self.subGtrainNodesNUM]
            batchlen = self.subGtrainNodesNUM - offset
            cacheLabel = self.nodeLabels[sampleIDs[0:self.subGtrainNodesNUM - offset]]
        
        sampleTime = time.time()
        logger.debug("sampleIDs shape:{}".format(len(sampleIDs)))
        #self.sampleNeig(sampleIDs,cacheGraph)
        self.sampleNeigGPU(sampleIDs,cacheGraph,self.cudaDevice)
        logger.debug("cacheGraph shape:{}, first graph shape:{}".format(len(cacheGraph),len(cacheGraph[0][0])))
        logger.info("sample subG cost {}s".format(time.time()-sampleTime))

        featTime = time.time()
        cacheFeat = self.featMerge(cacheGraph)
        logger.debug("featLen shape:{}".format(cacheFeat.shape))
        logger.info("subG feat merge cost {}s".format(time.time()-featTime))

        transTime = time.time()
        if self.framework == "dgl":
            cacheGraph = self.transGraph2DGLBlock(cacheGraph)
        elif self.framework == "pyg":
            cacheGraph = self.transGraph2PYGBatch(cacheGraph)
        logger.debug("subG trans cost {}s".format(time.time()-transTime))

        cacheData = [cacheGraph,cacheFeat,cacheLabel,batchlen]
        self.graphPipe.put(cacheData)
        self.trainptr += 1
        return 0
    
    def preGPUBatch(self):
        # 迁移到GPU中
        # 1-hop 采样
        
        # 2-hop 采样

        # 3-hop 采样
        
        pass

########################## 特征提取 ##########################
    def loadingFeatFileHead(self):
        for index in range(self.partNUM):
            filePath = self.dataPath + "/part" + str(index)
            file = open(filePath+"/feat.bin", "r+b")
            self.readfile.append(file)
            self.mmapfile.append(mmap.mmap(self.readfile[-1].fileno(), 0, access=mmap.ACCESS_DEFAULT))
        logger.info("mmap file success...")

    def closeMMapFileHead(self):
        for file in self.mmapfile:
            file.close()
        for file in self.readfile:
            file.close()

    def loadingMemFeat(self,rank):
        filePath = self.dataPath + "/part" + str(rank)
        tmp_feat = np.fromfile(filePath+"/feat.bin", dtype=np.float32)
        if self.feats == []:
            self.feats = torch.from_numpy(tmp_feat).reshape(-1,100)
        else:
            tmp_feat = torch.from_numpy(tmp_feat).reshape(-1,100)
            self.feats = torch.cat([self.feats,tmp_feat])
    
    def featMerge(self,cacheGraph):    
        nodeids = cacheGraph[0][0]
        nodeids = torch.cat([torch.tensor([0]),nodeids.to(device='cpu')])
        return self.feats[nodeids]

########################## 数据调整 ##########################
    def cleanPipe(self):
        # 清理数据管道及信号
        while self.graphPipe.qsize() > 0:
            self.graphPipe.get()    
        while self.sampleFlagQueue.qsize() > 0:
            self.sampleFlagQueue.get()

    def changeMode(self,mode):
        logger.info("change mode from:'{}' to '{}'...".format(self.mode,mode))
        # 数据集模式:[训练状态，验证状态，测试状态]
        # 1.修改训练模式
        lastMode = self.mode
        self.mode = mode
        # 2.清空管道与信号量
        self.cleanPipe()
        # 3.加载新训练节点
        self.cleanLastModeData(lastMode)
        self.loadModeData(self.mode)
        
        # 4.重置并初始化数据
        self.cacheData = [] 
        self.feats == []
        self.trainSubGTrack = self.randomTrainList()    
        self.subGptr = -1
        self.loadingGraph(merge=False)
        self.loadingMemFeat(self.trainSubGTrack[self.subGptr//self.partNUM][self.subGptr%self.partNUM])
        self.initNextGraphData()
        self.sampleFlagQueue.put(self.executor.submit(self.preGraphBatch)) #发送采样命令                          
    
    def cleanLastModeData(self,mode):
        logger.info("clean last mode:'{}' data".format(mode))
        if mode == "train":
            self.trainNodeDict = {}
            self.trainNodeNumbers = 0
        elif mode == "val":
            self.valNodeDict = {}
            self.valNodeNumbers = 0
        elif mode == "test":
            self.testNodeDict = {}
            self.testNodeNumbers = 0
    
    def loadModeData(self,mode):
        logger.info("loading mode:'{}' data".format(mode))
        if "train" == mode:
            self.trainNodeDict,self.trainNodeNumbers = self.loadingTrainID() # 训练节点字典，训练节点数目
            self.NodeLen = self.trainNUM
        elif "val" == mode:
            self.valNodeDict,self.valNodeNumbers = self.loadingValID() # 训练节点字典，训练节点数目
            self.NodeLen = self.valNUM
        elif "test" == mode:
            self.testNodeDict,self.testNodeNumbers = self.loadingTestID() # 训练节点字典，训练节点数目
            self.NodeLen = self.testNUM

    def checkMode(self):
        print("now dataset mode is {}".format(self.mode))

    def modeReset(self):
        "重置加载状态"
        logger.info("reset mode {}...".format(self.mode))
        # 清空管道与信号量
        self.cleanPipe()
        
        # 重置并初始化数据
        self.cacheData = [] 
        self.feats == []
        self.trainSubGTrack = self.randomTrainList()    
        self.subGptr = -1
        self.loadingGraph(merge=False)
        self.loadingMemFeat(self.trainSubGTrack[self.subGptr//self.partNUM][self.subGptr%self.partNUM])
        self.initNextGraphData()
        self.sampleFlagQueue.put(self.executor.submit(self.preGraphBatch)) #发送采样命令

########################## dgl接口 ##########################
    def genBlockTemplate(self):
        template = []
        blocks = []
        ptr = 0
        seeds = [i for i in range(1,self.batchsize+1)]
        for number in self.fanout:
            dst = copy.deepcopy(seeds)
            src = copy.deepcopy(seeds)
            ptr = len(src) + 1    
            for ids in seeds:
                for i in range(number-1):
                    dst.append(ids)
                    src.append(ptr)
                    ptr += 1
            seeds = copy.deepcopy(src)
            # src.append(0)
            # dst.append(0)
            template.insert(0,[torch.tensor(src),torch.tensor(dst)])
        return template
        
    def transGraph2DGLBlock(self,graphdata):
        # 先生成掩码
        masks = []
        blocks = []
        for src, dst in graphdata:
            layer_mask = torch.ge(src, 0)
            masks.append(layer_mask)
            logger.debug("masks shape:{}".format(layer_mask.shape))
            #layer_mask = torch.cat((layer_mask, torch.tensor([True])))            
        template = copy.deepcopy(self.templateBlock)

        for index,mask in enumerate(masks):
            src,dst = template[index]
            src = src.to(device=('cuda:%d'%self.cudaDevice))
            dst = dst.to(device=('cuda:%d'%self.cudaDevice))
            src *= mask
            dst *= mask
        
        ## 修改部分
        for index,(src,dst) in enumerate(template):
            data = (src,dst)
            block = self.create_dgl_block(data,len(self.templateBlock[index][0])+1,(len(self.templateBlock[index][0])//self.fanout[-(index+1)])+1)
            blocks.append(block)
        return blocks

    def create_dgl_block(self, data, num_src_nodes, num_dst_nodes):
        row, col = data
        gidx = dgl.heterograph_index.create_unitgraph_from_coo(2, num_src_nodes, num_dst_nodes, row, col, 'coo')
        g = DGLBlock(gidx, (['_N'], ['_N']), ['_E'])
        return g

    def tmp_create_dgl_block(self,cacheData):
        blocks = []
        # 传入数据结构:二维list数组，分别为每一个hop的COO图数据
        # 输出时，最边缘图在前面
        for info in cacheData:
            # info 是每一层的图数据信息
            src = np.array(info[0],dtype=np.int32)
            dst = np.array(info[1],dtype=np.int32)
            block = dgl.graph((src, dst))
            block = dgl.to_block(block)
            blocks.insert(0,block)  
        return blocks

########################## pyg接口 ##########################
    def genPYGBatchTemplate(self):
        zeros = torch.tensor([0])
        template_src = torch.empty(0,dtype=torch.int64)
        template_dst = torch.empty(0,dtype=torch.int64)
        ptr = self.batchsize + 1
        seeds = [i for i in range(1, self.batchsize + 1)]
        for number in self.fanout:
            dst = copy.deepcopy(seeds)
            src = copy.deepcopy(seeds)
            for ids in seeds:
                for i in range(number-1):
                    dst.append(ids)
                    src.append(ptr)
                    ptr += 1
            seeds = copy.deepcopy(src)
            template_src = torch.cat([template_src,torch.tensor(src,dtype=torch.int64)])
            template_dst = torch.cat([template_dst,torch.tensor(dst,dtype=torch.int64)])
        # template_src = torch.cat([template_src,zeros])
        # template_dst = torch.cat([template_dst,zeros])
        PYGTemplate = torch.stack([template_src,template_dst])
        return PYGTemplate

    def transGraph2PYGBatch(self,graphdata):
        # 先生成掩码
        masks = []
        for src, dst in graphdata:
            layer_mask = torch.ge(src, 0)
            masks.append(layer_mask)
        
        masks = torch.cat(masks)
        template = copy.deepcopy(self.templateBlock)
        template = template * masks
        return template

    def get_test_idx(self,testSize):
        #import random
        #part = random.randint(0,self.partNUM-1)
        #bound = self.idbound[part]
        np.random.seed()  # 设置随机种子，保证每次运行结果相同
        matrix = np.random.choice(range(0,self.idbound[self.partNUM-1][1]), size=testSize, replace=False)
        matrix = matrix.reshape(1,testSize)
        return torch.tensor(matrix).to(device='cpu')


def collate_fn(data):
    """
    data 输入结构介绍：
        [graph,feat]
    """
    return data[0]


if __name__ == "__main__":
    dataset = CustomDataset("./graphsage.json")
    with open("./graphsage.json", 'r') as f:
        config = json.load(f)
        batchsize = config['batchsize']
        epoch = config['epoch']
    train_loader = DataLoader(dataset=dataset, batch_size=batchsize, collate_fn=collate_fn,pin_memory=True)
    #time.sleep(2)
    for index in range(20):
        #start = time.time()
        for graph,feat,label,number in train_loader:
            #print("block:",graph)
            pass