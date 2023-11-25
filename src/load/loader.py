import concurrent.futures
from torch.utils.data import Dataset, DataLoader
from queue import Queue
import numpy as np
import json
import time
import dgl
import torch
from dgl.heterograph import DGLBlock
import copy
import sys
import logging
import os
import gc
import psutil
from tools import *
from memory_profiler import profile
#torch.set_printoptions(threshold=10000)
curFilePath = os.path.abspath(__file__)
curDir = os.path.dirname(curFilePath)

# 禁用操作
#logging.disable(logging.CRITICAL)
logging.basicConfig(level=logging.INFO,filename=curDir+'/../../log/loader.log',filemode='w',
                    format='%(message)s',datefmt='%H:%M:%S')
                    #format='%(message)s')
logger = logging.getLogger(__name__)

"""
数据加载的逻辑:#@profile
    1.生成训练随机序列
    2.预加载训练节点(所有的训练节点都被加载进入)
    2.预加载图集合
    3.不断生成采样子图
    4.当图采样完成后释放当前子图,加载下一个图
"""
class CustomDataset(Dataset):
    def __init__(self,confPath):
        self.cacheData = []  # 子图存储部分
        self.indptr = [] # dst / bound
        self.indices = [] # src / edgelist
        self.graphPipe = Queue()  # 采样存储管道   
        self.lossG = False

        self.preFetchExecutor = concurrent.futures.ThreadPoolExecutor(1)  # 线程池
        self.preFetchFlagQueue = Queue()
        self.preFetchDataCache = Queue()
        self.prefeat = 0

        #### 系统部分
        gpu = torch.cuda.get_device_properties(0) # use cuda:0
        self.gpumem = int(gpu.total_memory)

        #### config json 部分 ####
        self.dataPath = ''
        self.batchsize,self.cacheNUM,self.partNUM = 0,0,0
        self.maxEpoch,self.classes,self.epochInterval = 0,0,0
        self.featlen = 0
        self.fanout = []
        self.maxPartNodeNUM,self.mem = 0,0
        self.edgecut, self.nodecut,self.featDevice = 0,0,""
        self.train_name,self.framework,self.mode,self.dataset = "","","",""
        self.readTrainConfig(confPath)  # 训练参数加载
        # ================
        self.datasetInfo = self.readDatasetInfo()


        #### 训练记录 ####
        self.trainSubGTrack = self.randomTrainList()    # 训练轨迹
        self.subGptr = -1  # 子图训练指针，记录当前训练的位置，在加载图时发生改变
        
        #### 节点类型加载 ####
        self.NodeLen = 0        # 用于记录数据集中节点的数目，默认为train节点个数
        self.trainNUM = 0       # 训练集总数目
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
        self.trainptr = 0               # 当前训练集读取位置
        self.trainLoop = 0              # 当前子图可读取次数      
        self.lossMap = []
        #### mmap 特征部分 ####
        self.feats = torch.zeros([self.maxPartNodeNUM, self.featlen], dtype=torch.float32).to(self.featDevice)  ## 这一步会爆显存
        #self.feats = []
        self.addFeatMap = []
        self.memfeat = []

        #### 数据预取 ####
        self.template_cache_graph , self.ramapNodeTable = self.initCacheData() # CPU , GPU
        self.initNextGraphData()
        self.uniTable = torch.zeros(len(self.ramapNodeTable),dtype=torch.int32).cuda()


    def __len__(self):
        return self.NodeLen
    
    def __getitem__(self, index):
        if index % self.batchsize == 0:
            self.preGraphBatch()
            cacheData = self.graphPipe.get()
            return tuple(cacheData[:4])

########################## 初始化训练数据 ##########################
    def readTrainConfig(self,confPath):
        with open(confPath, 'r') as f:
            config = json.load(f)
        self.train_name = config['train_name']
        self.dataPath = config['datasetpath']+"/"+config['dataset']
        self.dataset = config['dataset']
        self.batchsize = config['batchsize']
        self.cacheNUM = config['cacheNUM']
        self.partNUM = config['partNUM']
        self.maxEpoch = config['maxEpoch']
        self.featlen = config['featlen']
        self.fanout = config['fanout']
        self.framework = config['framework']
        self.mode = config['mode']
        self.classes = config['classes']
        self.epochInterval = config['epochInterval']
        self.mem = config['memUse']
        self.maxPartNodeNUM = config['maxPartNodeNUM']
        self.edgecut = config['edgecut']
        self.nodecut = config['nodecut']
        self.featDevice = config['featDevice']

    def readDatasetInfo(self):
        confPath = self.dataPath + f"/{self.dataset}.json"
        with open(confPath, 'r') as f:
            config = json.load(f)
        return config

    def randomTrainList(self): 
        epochList = []
        for _ in range(self.maxEpoch + 1): # 额外多增加一行
            tarinArray = np.array(self.datasetInfo["path"])
            epochList.append(tarinArray)
        return epochList

########################## 加载/释放 图结构数据 ##########################
    #@profile
    def initNextGraphData(self):
        # 先拿到本次加载的内容，然后发送预取命令
        logger.info("----------initNextGraphData----------")
        start = time.time()
        self.subGptr += 1
        self.GID = self.trainSubGTrack[self.subGptr // self.partNUM][self.subGptr % self.partNUM]
        print(f"loading G :{self.GID}..")
        if self.subGptr == 0:
            self.loadingGraphData(self.GID) # 第一个需要从头加载
        else:
            taskFlag = self.preFetchFlagQueue.get()
            taskFlag.result()
            preCacheData = self.preFetchDataCache.get()
            self.loadingGraphData(self.GID,predata=preCacheData)
        self.trainNodes = self.trainNodeDict[self.GID]
        self.subGtrainNodesNUM = self.trainNodeNumbers[self.GID]   
        self.trainLoop = ((self.subGtrainNodesNUM - 1) // self.batchsize) + 1
        self.preFetchFlagQueue.put(self.preFetchExecutor.submit(self.preloadingGraphData))  # 发送下一个预取命令
        logger.info(f"loading next graph with {time.time() - start :.4f}s")

    def loadingTrainID(self):
        # 加载子图所有训练集
        idDict = {}
        numberList = [0 for i in range(self.partNUM)]  
        for index in range(self.partNUM):
            filePath = self.dataPath + "/part" + str(index)   
            trainIDs = torch.as_tensor(np.fromfile(filePath+"/trainIds.bin",dtype=np.int64))
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
        start = time.time()
        ptr = self.subGptr + 1
        rank = self.trainSubGTrack[ptr // self.partNUM][ptr % self.partNUM]
        filePath = self.dataPath + "/part" + str(rank)
        indices = np.fromfile(filePath + "/indices.bin", dtype=np.int32)
        indptr = np.fromfile(filePath + "/indptr.bin", dtype=np.int32)
        nodeLabels = np.fromfile(filePath+"/labels.bin", dtype=np.int64)

        ###
        # 增量特征加载
        map = self.addFeatMap
        sameNodeInfoPath = filePath + '/sameNodeInfo.bin'
        diffNodeInfoPath = filePath + '/diffNodeInfo.bin'
        sameNode = torch.as_tensor(np.fromfile(sameNodeInfoPath, dtype = np.int32))
        diffNode = torch.as_tensor(np.fromfile(diffNodeInfoPath, dtype = np.int32))
        res1_one, res2_one = torch.split(sameNode, (sameNode.shape[0] // 2))
        
        newMap = torch.clone(map)
        newMap[res2_one.to(torch.int64)] = map[res1_one.to(torch.int64)]
        if diffNode.shape[0] != 0:
            res1_zero, res2_zero = torch.split(diffNode, (diffNode.shape[0] // 2))
        else:
            res1_zero,res2_zero = torch.Tensor([]),torch.Tensor([])
        addFeat = torch.as_tensor(np.fromfile(filePath + "/addfeat.bin", dtype=np.float32).reshape(-1, self.featlen))
        replace_idx = map[res1_zero[:addFeat.shape[0]].to(torch.int64)].to(torch.int64)
        newMap[res2_zero.to(torch.int64)] = map[res1_zero.to(torch.int64)]
        addFeatInfo = {"addFeat": addFeat, "replace_idx": replace_idx, "map": newMap} 
        self.preFetchDataCache.put([indices,indptr,addFeatInfo,nodeLabels])
        print(f"pre data time :{time.time() - start:.4f}s...")
        return 0

    #@profile
    def loadingGraphData(self,subGID,predata=None):
        filePath = self.dataPath + "/part" + str(subGID)
        if predata == None:
            # 第一次初始化全加载
            self.indices = torch.as_tensor(np.fromfile(filePath + "/indices.bin", dtype=np.int32))
            self.indptr = torch.as_tensor(np.fromfile(filePath + "/indptr.bin", dtype=np.int32))
            self.nodeLabels = torch.as_tensor(np.fromfile(filePath + "/labels.bin", dtype=np.int64))
            addFeat = np.fromfile(filePath + "/feat.bin", dtype=np.float32).reshape(-1, self.featlen)
            self.addFeatMap = torch.arange(self.maxPartNodeNUM, dtype=torch.int64).to(self.featDevice)  
        else:
            # 预加载完成，进行数据处理，此时的预取数据都保持在CPU中
            self.indices = torch.as_tensor(predata[0])
            self.indptr = torch.as_tensor(predata[1])
            self.nodeLabels = torch.as_tensor(predata[3])
            addFeatInfo = predata[2]
            addFeat = addFeatInfo['addFeat']#.to(self.featDevice)
            self.addFeatMap = addFeatInfo['map'].to(self.featDevice)  
            replace_idx = addFeatInfo['replace_idx']#.to(self.featDevice)
             
        # 判断是否裁剪，之后放入GPU
        graphNodeNUM,graphEdgeNUM = int(len(self.indptr) - 1 ),len(self.indices)
        if countMemToLoss(graphEdgeNUM,graphNodeNUM,self.featlen,self.mem):
            self.lossG = True   # 需要裁剪
            sortNode = torch.as_tensor(np.fromfile(filePath + "/sortIds.bin", dtype=np.int32))
            saveRatio = 0.99
            cutNode,saveNode = sortNode[int(graphNodeNUM*saveRatio):],sortNode[:int(graphNodeNUM*saveRatio)]
            start = time.time()
            self.indptr,self.indices,self.lossMap = \
                streamLossGraph(self.indptr,self.indices,cutNode,sliceNUM=4,randomLoss=0,degreeCut=None,CutRatio=0.5)
            # print(f"loss_csr time :{time.time() - start:.4f}s...")
            start = time.time()
            # addFeat -> self.feat device位置 
            sliceFeatNUM = 8
            if predata == None: 
                # 表明首次加载,直接迁移
                loss_feat(self.feats, addFeat , sliceFeatNUM, self.lossMap, self.featlen, self.featDevice)
                self.memfeat = torch.as_tensor(addFeat)
            else:
                self.memfeat[replace_idx] = addFeat     #内存feat进行替换，此时replace_idx已经做过map映射
                loss_feat(self.feats, self.memfeat , sliceFeatNUM, self.lossMap, self.featlen, self.featDevice)
                print(f"loading feat time :{time.time() - start:.4f}s...")
        else:
            # 不需要进行裁剪,csr,feat,label直接存入cuda
            print("not need cut ...")
            self.lossG = False 
            self.indptr,self.indices = self.indptr.cuda(),self.indices.cuda()
            if predata == None: # 表明首次加载,直接迁移
                idx = torch.arange(addFeat.shape[0],dtype=torch.int64,device="cuda")
                addFeat = torch.as_tensor(addFeat)
                streamAssign(self.feats,idx,addFeat,sliceNUM=4)
            else:
                # 流式处理
                streamAssign(self.feats,replace_idx,addFeat,sliceNUM=4)


########################## 采样图结构 ##########################
    def sampleNeigGPU_NC(self,sampleIDs,cacheGraph,batchlen):     
        logger.info("----------[sampleNeigGPU_NC]----------")
        sampleIDs = sampleIDs.to(torch.int32).to('cuda:0')
        sampleStart = time.time()
        ptr,seedPtr,NUM = 0, 0, 0
        mapping_ptr = [ptr]
        for l, fan_num in enumerate(self.fanout):
            if l == 0:
                seed_num = batchlen
            else:
                seed_num = len(sampleIDs)
            self.ramapNodeTable[seedPtr:seedPtr+seed_num] = sampleIDs
            seedPtr += seed_num
            out_src = cacheGraph[0][ptr:ptr+seed_num*fan_num]
            out_dst = cacheGraph[1][ptr:ptr+seed_num*fan_num]
            
            if self.lossG == False:
                NUM = dgl.sampling.sample_with_edge(self.indptr,self.indices,
                    sampleIDs,seed_num,fan_num,out_src,out_dst)
            elif self.lossG == True:
                NUM = dgl.sampling.sample_with_edge_and_map(self.indptr,self.indices,
                    sampleIDs,seed_num,fan_num,out_src,out_dst,self.lossMap)
            sampleIDs = cacheGraph[0][ptr:ptr+NUM.item()]
            ptr=ptr+NUM.item()
            mapping_ptr.append(ptr)
        self.ramapNodeTable[seedPtr:seedPtr+NUM] = sampleIDs
        seedPtr += NUM 
        logger.info("Sample Neighbor Time {:.5f}s".format(time.time()-sampleStart))
        mappingTime = time.time()        
        cacheGraph[0] = cacheGraph[0][:mapping_ptr[-1]]
        cacheGraph[1] = cacheGraph[1][:mapping_ptr[-1]]
        unique = self.uniTable.clone()
        logger.info("construct remapping data Time {:.5f}s".format(time.time()-mappingTime))
        
        t = time.time()  
        #cacheGraph[0],cacheGraph[1],unique = dgl.remappingNode(cacheGraph[0],cacheGraph[1],unique)
        
        cacheGraph[0],cacheGraph[1],unique = dgl.mapByNodeSet(self.ramapNodeTable[:seedPtr],unique,cacheGraph[0],cacheGraph[1])
        # print("min idx",torch.min(self.lossMap[unique.to(torch.int64)]))
        # print("max map idx",torch.max(self.lossMap))
        # print("self.feats:",self.feats.shape)
        # print("unique:",unique)
        # print("unique.shape:",unique.shape)
        logger.info("cuda remapping func Time {:.5f}s".format(time.time()-t))
        transTime = time.time()
        if self.framework == "dgl":
            layerNUM = len(mapping_ptr) - 1
            blocks = []
            dstNUM, srcNUM = 0, 0
            for layer in range(1,layerNUM+1):
                src = cacheGraph[0][:mapping_ptr[layer]]
                dst = cacheGraph[1][:mapping_ptr[layer]]
                data = (src,dst)
                if layer == 1:
                    dstNUM,_ = torch.max(dst,dim=0)
                    srcNUM,_ = torch.max(src,dim=0)
                    dstNUM += 1
                    srcNUM += 1      
                elif layer == layerNUM:
                    dstNUM = srcNUM
                    srcNUM = len(unique)
                else:
                    dstNUM = srcNUM
                    srcNUM,_ = torch.max(src,dim=0)
                    srcNUM += 1
                block = self.create_dgl_block(data,srcNUM,dstNUM)
                blocks.insert(0,block)
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
        remapTable = []
        for _, fan in enumerate(self.fanout):
            dst = torch.full((tmp * fan,), -1, dtype=torch.int32).cuda()  # 使用PyTorch张量，指定dtype
            src = torch.full((tmp * fan,), -1, dtype=torch.int32).cuda()  # 使用PyTorch张量，指定dtype
            cacheGraph[0].append(src)
            cacheGraph[1].append(dst)
            tmp = tmp * (fan + 1)
        remapTable = copy.deepcopy(cacheGraph[0])
        remapTable.append(cacheGraph[1][-1])
        remapTable = torch.cat(remapTable,dim=0).to(torch.int32).cuda()
        cacheGraph[0] = torch.cat(cacheGraph[0],dim=0)
        cacheGraph[1] = torch.cat(cacheGraph[1],dim=0)
        return cacheGraph ,remapTable

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
        blocks,uniqueList = self.sampleNeigGPU_NC(sampleIDs,cacheGraph,batchlen)
        logger.info("sample subG all cost {:.5f}s".format(time.time()-sampleTime))
        ##
     
        featTime = time.time()
        cacheFeat = self.featMerge(uniqueList)
        logger.info("feat merge cost {:.5f}s".format(time.time()-featTime))
        
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
        if self.lossG == False:
            featIdx = self.addFeatMap[uniqueList.to(torch.int64).to(self.feats.device)]
            test = self.feats[featIdx]
        elif self.lossG == True:
            mapIdx = self.lossMap[uniqueList.to(self.lossMap.device).to(torch.int64)].to(torch.int64)     
            test = self.feats[mapIdx.to(self.feats.device)]        
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
    dataset = CustomDataset(curDir+"/../../config/PA.json")
    with open(curDir+"/../../config/PA.json", 'r') as f:
        config = json.load(f)
        batchsize = config['batchsize']
        epoch = config['maxEpoch']
    train_loader = DataLoader(dataset=dataset, batch_size=batchsize,collate_fn=collate_fn)
    count = 0
    for index in range(2):
        start = time.time()
        loopTime = time.time()
        for graph,feat,label,number in train_loader:
            # print(graph)
            # print(feat.shape)
            # print(label)
            # print(number)
            # print('-'*20)
            count = count + 1
            # if count % 20 == 0:
            #     print("loop time:{:.5f}".format(time.time()-loopTime))
        print("="*20)
        print("all loop time:{:.5f}".format(time.time()-loopTime))
        print("="*20)