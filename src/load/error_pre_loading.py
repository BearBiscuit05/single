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
# from memory_profiler import profile

# logging.basicConfig(level=logging.INFO,filename='../../log/loader.log',filemode='w',
#                     format='%(asctime)s-%(levelname)s-%(message)s',datefmt='%H:%M:%S')
# format='%(message)s')
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
logging.basicConfig(level=logging.INFO, filename='./loader.log', filemode='w',
                    format='%(message)s', datefmt='%H:%M:%S')
# format='%(message)s')
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
    # @profile(precision=4, stream=open('./__init__.log','w+'))
    def __init__(self, confPath, pre_fetch=False):
        self.pre_fetch = pre_fetch

        #### 采样资源 ####
        self.cacheData = []  # 子图存储部分
        self.graphPipe = Queue()  # 采样存储管道
        self.sampleFlagQueue = Queue()
        self.executor = concurrent.futures.ThreadPoolExecutor(1)  # 线程池

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
        self.epoch,self.preRating,self.classes = 0,0,0
        self.featlen = 0
        self.idbound,self.fanout = [],[]
        self.train_name,self.framework,self.mode,self.dataset = "","","",""
        self.readConfig(confPath)
        # ================

        #### 训练记录 ####
        self.trainSubGTrack = self.randomTrainList()  # 训练轨迹
        self.subGptr = -1  # 子图训练指针，记录当前训练的位置，在加载图时发生改变

        #### 节点类型加载 ####
        self.NodeLen = 0  # 用于记录数据集中节点的数目，默认为train节点个数
        self.trainNUM = 0  # 训练集总数目
        self.valNUM = 0
        self.testNUM = 0
        self.trainNodeDict, self.valNodeDict, self.testNodeDict = {}, {}, {}
        self.trainNodeNumbers, self.valNodeNumbers, self.testNodeNumbers = 0, 0, 0
        self.loadModeData(self.mode)

        #### 图结构信息 ####
        self.graphNodeNUM = 0  # 当前训练子图节点数目
        self.graphEdgeNUM = 0  # 当前训练子图边数目
        self.trainingGID = 0  # 当前训练子图的ID
        self.subGtrainNodesNUM = 0  # 当前训练子图训练节点数目
        self.trainNodes = []  # 训练子图训练节点记录
        self.nodeLabels = []  # 子图标签
        self.nextGID = 0  # 下一个训练子图
        self.trainptr = 0  # 当前训练集读取位置
        self.trainLoop = 0  # 当前子图可读取次数
        #### mmap 特征部分 ####
        self.readfile = []  # 包含两个句柄/可能有三个句柄
        self.mmapfile = []
        self.feats = []

        #### 规定用哪张卡单独跑 ####
        self.cudaDevice = 0

        #### 数据预取 ####
        self.template_cache_graph, self.template_cache_label = self.initCacheData()
        self.loadingGraph(merge=False)
        self.loadingMemFeat(self.trainSubGTrack[self.subGptr // self.partNUM][self.subGptr % self.partNUM])
        self.initNextGraphData()

    def __len__(self):
        return self.NodeLen

    def __getitem__(self, index):
        # 批数据预取 缓存1个
        if index % self.preRating == 0:
            self.sampleFlagQueue.put(self.executor.submit(self.preGraphBatch))

        # 获取采样数据
        if index % self.batchsize == 0:
            # 调用实际数据
            if self.graphPipe.qsize() > 0:
                self.sampleFlagQueue.get()
                cacheData = self.graphPipe.get()
                if self.train_name == "LP":
                    return cacheData[0], cacheData[1], cacheData[2], cacheData[3], cacheData[4], cacheData[5]
                else:
                    return cacheData[0], cacheData[1], cacheData[2], cacheData[3]
            else:  # 需要等待
                flag = self.sampleFlagQueue.get()
                data = flag.result()
                cacheData = self.graphPipe.get()
                if self.train_name == "LP":
                    return cacheData[0], cacheData[1], cacheData[2], cacheData[3], cacheData[4], cacheData[5]
                else:
                    return cacheData[0], cacheData[1], cacheData[2], cacheData[3]
        return 0, 0

        # if index % self.batchsize == 0:
        #     self.preGraphBatch()
        #     cacheData = self.graphPipe.get()
        #     return cacheData[0], cacheData[1], cacheData[2], cacheData[3]
        # return 0, 0


    ########################## 初始化训练数据 ##########################
    def readConfig(self, confPath):
        with open(confPath, 'r') as f:
            config = json.load(f)
        self.train_name = config['train_name']
        self.dataPath = config['datasetpath'] + "/" + config['dataset']
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
        # print(formatted_data)

    def custom_sort(self):
        idMap = {}
        for i in range(self.partNUM):
            folder_path = self.dataPath + "/part" + str(i)
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
            for idx in range(0, self.partNUM):
                if idx == 0:
                    num = lastid
                else:
                    num = tmp[-1]
                candidates = idMap[num]
                available_candidates = [int(candidate) for candidate in candidates if
                                        int(candidate) not in used_numbers]
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
        # epochList = self.custom_sort()
        epochList = []
        for i in range(self.epoch + 1):  # 额外多增加一行
            random_array = np.random.choice(np.arange(0, self.partNUM), size=self.partNUM, replace=False)
            if len(epochList) == 0:
                epochList.append(random_array)
            else:
                # 已经存在列
                lastid = epochList[-1][-1]
                while (lastid == random_array[0]):
                    random_array = np.random.choice(np.arange(0, self.partNUM), size=self.partNUM, replace=False)
                epochList.append(random_array)

        logger.info("train track:{}".format(epochList))
        return epochList

    ########################## 加载/释放 图结构数据 ##########################
    # @profile(precision=4, stream=open('./info.log','w+'))
    def initNextGraphData(self):
        start = time.time()
        # 查看是否需要释放
        if self.subGptr > 0:
            print("moving...")
            self.moveGraph()
            print("after moving feats:", self.feats)
        # 对于将要计算的子图(已经加载)，修改相关信息
        self.trainingGID = self.trainSubGTrack[self.subGptr // self.partNUM][self.subGptr % self.partNUM]
        self.graphNodeNUM = int(len(self.cacheData[1]) / 2)  # 获取当前节点数目
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
        
        # TODO:此处合并
        # 加载数据预取结果
        prefetch = False
        if self.preFetchFlagQueue.qsize() > 0:
        # if self.preFetchDataCache.qsize() > 0:
            prefetch = True
            taskFlag = self.preFetchFlagQueue.get()
            flag = taskFlag.result()
            print("flag:=======> ",flag)
            preCacheData = self.preFetchDataCache.get()   
            # self.loadingGraph(preFetch=True)
            self.loadingGraph(preFetch=True,srcList=preCacheData[0],rangeList=preCacheData[1])
        else:
            self.loadingGraph()
        
        self.nextGID = self.trainSubGTrack[self.subGptr // self.partNUM][self.subGptr % self.partNUM]
        halostart = time.time()
        self.loadingHalo()
        logger.info("loadingHalo time: %g" % (time.time() - halostart))
        haloend = time.time()
        
        if prefetch:
            self.loadingMemFeat(self.nextGID,preFetch=True,preFeat=preCacheData[2])
            prefetch = False
        else:
            self.loadingMemFeat(self.nextGID)

        logger.info("loadingHalo time: %g" % (time.time() - haloend))
        logger.info("当前加载图为:{},下一个图:{},图训练集规模:{},图节点数目:{},图边数目:{},加载耗时:{:.5f}s" \
                    .format(self.trainingGID, self.nextGID, self.subGtrainNodesNUM, \
                            self.graphNodeNUM, self.graphEdgeNUM, time.time() - start))

        # 发送预取命令,存储在指定位置,(此处数据的预加载已经完成)
        self.preFetchFlagQueue.put(self.preFetchExecutor.submit(self.preloadingGraphData))
        print("self.preFetchFlagQueue.qsize() :", self.preFetchFlagQueue.qsize())

    def loadingTrainID(self):
        # 加载子图所有训练集
        idDict = {}
        numberList = [0 for i in range(self.partNUM)]
        for index in range(self.partNUM):
            filePath = self.dataPath + "/part" + str(index)
            trainIDs = torch.load(filePath + "/trainID.bin")
            # trainIDs = trainIDs.to(torch.uint8).nonzero().squeeze()[:TESTNODE]
            trainIDs = trainIDs.to(torch.uint8).nonzero().squeeze()
            idDict[index], _ = torch.sort(trainIDs)
            current_length = len(idDict[index])
            numberList[index] = current_length
            fill_length = self.batchsize - current_length % self.batchsize
            padding = torch.full((fill_length,), -1, dtype=idDict[index].dtype)
            idDict[index] = torch.cat((idDict[index], padding))
            logger.debug("subG:{} ,real train len:{}, padding number:{}".format(index, current_length, padding))
            self.trainNUM += idDict[index].shape[0]
        return idDict, numberList

    def loadingValID(self):
        # 加载子图所有训练集
        idDict = {}
        numberList = [0 for i in range(self.partNUM)]
        for index in range(self.partNUM):
            filePath = self.dataPath + "/part" + str(index)
            ValIDs = torch.load(filePath + "/valID.bin")
            ValIDs = ValIDs.to(torch.uint8).nonzero().squeeze()
            idDict[index], _ = torch.sort(ValIDs)
            current_length = len(idDict[index])
            numberList[index] = current_length
            fill_length = self.batchsize - current_length % self.batchsize
            padding = torch.full((fill_length,), -1, dtype=idDict[index].dtype)
            idDict[index] = torch.cat((idDict[index], padding))
            self.valNUM += idDict[index].shape[0]
        return idDict, numberList

    def loadingTestID(self):
        # 加载子图所有训练集
        idDict = {}
        numberList = [0 for i in range(self.partNUM)]
        for index in range(self.partNUM):
            filePath = self.dataPath + "/part" + str(index)
            TestID = torch.load(filePath + "/testID.bin")
            TestID = TestID.to(torch.uint8).nonzero().squeeze()
            idDict[index], _ = torch.sort(TestID)
            current_length = len(idDict[index])
            numberList[index] = current_length
            fill_length = self.batchsize - current_length % self.batchsize
            padding = torch.full((fill_length,), -1, dtype=idDict[index].dtype)
            idDict[index] = torch.cat((idDict[index], padding))
            self.testNUM += idDict[index].shape[0]
        return idDict, numberList

    def preloadingGraphData(self):
        # 暂时只转换为numpy格式
        ptr = self.subGptr + 1
        rank = self.trainSubGTrack[ptr // self.partNUM][ptr % self.partNUM]
        filePath = self.dataPath + "/part" + str(rank)
        srcdata = np.fromfile(filePath + "/srcList.bin", dtype=np.int32)
        rangedata = np.fromfile(filePath + "/range.bin", dtype=np.int32)
        tmp_feat = np.fromfile(filePath + "/feat.bin", dtype=np.float32)
        self.preFetchDataCache.put([srcdata,rangedata,tmp_feat])
        return 0

    def loadingGraph(self, merge=True, preFetch=False,srcList=None,rangeList=None):
        """
        merge 表示此处需要融合两个图结构
        preFetch 表示下一需要融合的图已经被加载
        """
        # 加载下一个等待训练的图
        self.subGptr += 1
        subGID = self.trainSubGTrack[self.subGptr // self.partNUM][self.subGptr % self.partNUM]
        filePath = self.dataPath + "/part" + str(subGID)
        # TODO:存在优化/加速加载

        if preFetch == False:
            srcdata = np.fromfile(filePath + "/srcList.bin", dtype=np.int32)
            srcdata = torch.tensor(srcdata, device=('cuda:%d' % self.cudaDevice))
            rangedata = np.fromfile(filePath + "/range.bin", dtype=np.int32)
            rangedata = torch.tensor(rangedata, device=('cuda:%d' % self.cudaDevice))
            if merge:
                srcdata = srcdata + self.graphNodeNUM
                rangedata = rangedata + self.graphEdgeNUM
                # TODO:存在优化
                self.cacheData[0] = torch.cat([self.cacheData[0], srcdata])
                self.cacheData[1] = torch.cat([self.cacheData[1], rangedata])
            else:
                # TODO:存在优化
                self.cacheData.append(srcdata)
                self.cacheData.append(rangedata)

        else:
            srcList = torch.tensor(srcList, device=('cuda:%d' % self.cudaDevice))
            rangeList = torch.tensor(rangeList, device=('cuda:%d' % self.cudaDevice))
            srcList = srcList + self.graphNodeNUM
            rangeList = rangeList + self.graphEdgeNUM
            self.cacheData[0] = torch.cat([self.cacheData[0], srcList])
            self.cacheData[1] = torch.cat([self.cacheData[1], rangeList])

    def loadingLabels(self, rank):
        filePath = self.dataPath + "/part" + str(rank)
        if self.dataset == "papers100M_64":
            labels = torch.from_numpy(np.fromfile(filePath + "/label.bin", dtype=np.int32)).to(torch.int64)
        else:
            labels = torch.from_numpy(np.fromfile(filePath + "/label.bin", dtype=np.int32)).to(torch.int64)
        return labels

    def moveGraph(self):
        # TODO: 数据预取需要修改,在此处调整位置，删除前段位置，计算中段位置，重组中后段
        logger.debug("move last graph {},and now graph {}".format(self.trainingGID, self.nextGID))
        logger.debug("befor move srclist len:{}".format(len(self.cacheData[0])))
        logger.debug("befor move range len:{}".format(len(self.cacheData[1])))
        self.cacheData[0] = self.cacheData[0][self.graphEdgeNUM:]  # 边
        self.cacheData[1] = self.cacheData[1][self.graphNodeNUM * 2:]  # 范围
        self.cacheData[0] = self.cacheData[0] - self.graphNodeNUM  # 边 nodeID
        self.cacheData[1] = self.cacheData[1] - self.graphEdgeNUM
        self.feats = self.feats[self.graphNodeNUM:]
        logger.debug("after move srclist len:{}".format(len(self.cacheData[0])))
        logger.debug("after move range len:{}".format(len(self.cacheData[1])))
        gc.collect()

    def loadingHalo(self):
        # 要先加载下一个子图，然后再加载halo( 当前<->下一个 )
        filePath = self.dataPath + "/part" + str(self.trainingGID)
        deviceName = 'cuda:%d' % self.cudaDevice
        try:
            edges = np.fromfile(filePath + "/halo" + str(self.nextGID) + ".bin", dtype=np.int32)
            edges = torch.tensor(edges, device=deviceName, dtype=torch.int32).contiguous()
            bound = np.fromfile(filePath + "/halo" + str(self.nextGID) + "_bound.bin", dtype=np.int32)
            bound = torch.tensor(bound, device=deviceName, dtype=torch.int32).contiguous()
            self.cacheData[0] = self.cacheData[0].contiguous()
            self.cacheData[1] = self.cacheData[1].contiguous()
            signn.torch_graph_halo_merge(self.cacheData[0], self.cacheData[1], edges, bound, self.graphNodeNUM)
        except:
            logger.info("graph {} has no halo file with {}...".format(self.trainingGID, self.nextGID))
            print("graph {} has no halo file with {}...".format(self.trainingGID, self.nextGID))

    ########################## 采样图结构 ##########################
    def sampleNeigGPU_NC(self, sampleIDs, cacheGraph, batchlen):
        sampleIDs = sampleIDs.to(torch.int32).to('cuda:0')
        ptr = 0
        mapping_ptr = [ptr]
        batch = len(sampleIDs)
        sampleStart = time.time()
        for l, fan_num in enumerate(self.fanout):
            if l == 0:
                seed_num = batchlen
            else:
                seed_num = len(sampleIDs)
            out_src = cacheGraph[0][ptr:ptr + seed_num * fan_num]
            out_dst = cacheGraph[1][ptr:ptr + seed_num * fan_num]
            out_num = torch.Tensor([0]).to(torch.int64).to('cuda:0')
    
            signn.torch_sample_hop(
                self.cacheData[0], self.cacheData[1],
                sampleIDs, seed_num, fan_num,
                out_src, out_dst, out_num)
    
            # 得到dst + src  ==> 组合得到下一次的采样seed
            sampleIDs = cacheGraph[0][ptr:ptr + out_num.item()]
            ptr = ptr + out_num.item()
            mapping_ptr.append(ptr)
        logger.info("sample Time {:.5f}s".format(time.time() - sampleStart))
    
        mappingTime = time.time()
        cacheGraph[0] = cacheGraph[0][:mapping_ptr[-1]]
        cacheGraph[1] = cacheGraph[1][:mapping_ptr[-1]]
        uniqueNUM = torch.Tensor([0]).to(torch.int64).to('cuda:0')
        edgeNUM = mapping_ptr[-1]
        # TODO: 需要优化
        all_node = torch.cat([cacheGraph[1], cacheGraph[0]])
        unique = torch.zeros(mapping_ptr[-1] * 2, dtype=torch.int32).to('cuda:0')
        signn.torch_graph_mapping(all_node, cacheGraph[0], cacheGraph[1], cacheGraph[0], cacheGraph[1], unique, edgeNUM,
                                  uniqueNUM)
        unique = unique[:uniqueNUM.item()]
        logger.info("mapping Time {:.5f}s".format(time.time() - mappingTime))
    
        transTime = time.time()
    
        if self.framework == "dgl":
            layer = len(mapping_ptr) - 1
            blocks = []
            save_num = 0
            for index in range(1, layer + 1):
                src = cacheGraph[0][:mapping_ptr[-index]]
                dst = cacheGraph[1][:mapping_ptr[-index]]
                data = (src, dst)
                if index == 1:
                    save_num, _ = torch.max(dst, dim=0)
                    save_num += 1
                    block = self.create_dgl_block(data, uniqueNUM.item(), save_num)
                elif index == layer:
                    tmp_num = save_num
                    save_num, _ = torch.max(dst, dim=0)
                    save_num += 1
                    block = self.create_dgl_block(data, tmp_num, save_num)
                else:
                    tmp_num = save_num
                    save_num, _ = torch.max(dst, dim=0)
                    save_num += 1
                    block = self.create_dgl_block(data, tmp_num, save_num)
                blocks.append(block)
        elif self.framework == "pyg":
            src = cacheGraph[0][:mapping_ptr[-1]].to(torch.int64)
            dst = cacheGraph[1][:mapping_ptr[-1]].to(torch.int64)
            blocks = torch.stack((src, dst), dim=0)
        logger.info("trans Time {:.5f}s".format(time.time() - transTime))
        return blocks, unique

    def getNegNode(self, sampleIDs, batchlen, negNUM=1):
        sampleIDs = sampleIDs.to(torch.int32).to('cuda:0')
        out_src = torch.zeros(batchlen).to(torch.int32).to('cuda:0')
        out_dst = torch.zeros(batchlen).to(torch.int32).to('cuda:0')
        seed_num = batchlen
        fan_num = 1
        out_num = torch.Tensor([0]).to(torch.int64).to('cuda:0')
        signn.torch_sample_hop(
            self.cacheData[0][:self.graphEdgeNUM], self.cacheData[1][:self.graphNodeNUM * 2],
            sampleIDs, seed_num, fan_num,
            out_src, out_dst, out_num)
        # print("sample 1122")
        out_src = out_src[:out_num.item()]
        out_dst = out_dst[:out_num.item()]
        raw_src = copy.deepcopy(out_src)
        raw_dst = copy.deepcopy(out_dst)
        # print(raw_src.shape + raw_src.shape)
    
        neg_dst = torch.randint(low=0, high=self.graphNodeNUM, size=raw_src.shape).to(torch.int32).to("cuda:0")
    
        all_tensor = torch.cat([raw_src, raw_dst, raw_src, neg_dst])
        raw_edges = torch.cat([raw_src, raw_dst])
        src_cat = torch.cat([raw_src, raw_src])
        dst_cat = torch.cat([raw_dst, neg_dst])
        raw_src = copy.deepcopy(out_src)
        raw_dst = copy.deepcopy(out_dst)
        edgeNUM = len(src_cat)
        uniqueNUM = torch.Tensor([0]).to(torch.int64).to('cuda:0')
        unique = torch.zeros(len(all_tensor), dtype=torch.int32).to('cuda:0')
    
        signn.torch_graph_mapping(all_tensor, src_cat, dst_cat, src_cat, dst_cat, unique, edgeNUM, uniqueNUM)
        return unique[:uniqueNUM.item()], raw_edges, src_cat, dst_cat

    def sampleNeigGPU_LP(self, sampleIDs, raw_edges, cacheGraph, batchlen):
        sampleIDs = sampleIDs.to(torch.int32).to('cuda:0')
        ptr = 0
        mapping_ptr = [ptr]
        sampleStart = time.time()
        for l, fan_num in enumerate(self.fanout):
            if l == 0:
                seed_num = batchlen
            else:
                seed_num = len(sampleIDs)
            out_src = cacheGraph[0][ptr:ptr + seed_num * fan_num]
            out_dst = cacheGraph[1][ptr:ptr + seed_num * fan_num]
            out_num = torch.Tensor([0]).to(torch.int64).to('cuda:0')
            signn.torch_sample_hop(
                self.cacheData[0], self.cacheData[1],
                sampleIDs, seed_num, fan_num,
                out_src, out_dst, out_num)
    
            sampleIDs = cacheGraph[0][ptr:ptr + out_num.item()]
            ptr = ptr + out_num.item()
            mapping_ptr.append(ptr)
        logger.info("sample Time {:.5f}s".format(time.time() - sampleStart))
    
        mappingTime = time.time()
        cacheGraph[0] = cacheGraph[0][:mapping_ptr[-1]]
        cacheGraph[1] = cacheGraph[1][:mapping_ptr[-1]]
        uniqueNUM = torch.Tensor([0]).to(torch.int64).to('cuda:0')
        edgeNUM = mapping_ptr[-1]
        all_node = torch.cat([cacheGraph[1], cacheGraph[0]])
        unique = torch.zeros(mapping_ptr[-1] * 2, dtype=torch.int32).to('cuda:0')
        signn.torch_graph_mapping(all_node, cacheGraph[0], cacheGraph[1], cacheGraph[0], cacheGraph[1], unique, edgeNUM,
                                  uniqueNUM)
        unique = unique[:uniqueNUM.item()]
        logger.info("mapping Time {:.5f}s".format(time.time() - mappingTime))
    
        transTime = time.time()
    
        if self.framework == "dgl":
            layer = len(mapping_ptr) - 1
            blocks = []
            save_num = 0
            for index in range(1, layer + 1):
                src = cacheGraph[0][:mapping_ptr[-index]]
                dst = cacheGraph[1][:mapping_ptr[-index]]
                data = (src, dst)
                g = dgl.graph(data)
                block = dgl.to_block(g)
                blocks.append(block)
        elif self.framework == "pyg":
            src = cacheGraph[0][:mapping_ptr[-1]].to(torch.int64)
            dst = cacheGraph[1][:mapping_ptr[-1]].to(torch.int64)
            blocks = torch.stack((src, dst), dim=0)
        logger.info("trans Time {:.5f}s".format(time.time() - transTime))
        return blocks, unique

    def initCacheData(self):
        if self.train_name == "NC":
            number = self.batchsize
        else:
            number = self.batchsize * 3
        tmp = number
        cacheGraph = [[], []]
        for layer, fan in enumerate(self.fanout):
            dst = torch.full((tmp * fan,), -1, dtype=torch.int32).to("cuda:0")  # 使用PyTorch张量，指定dtype
            src = torch.full((tmp * fan,), -1, dtype=torch.int32).to("cuda:0")  # 使用PyTorch张量，指定dtype
            cacheGraph[0].append(src)
            cacheGraph[1].append(dst)
            tmp = tmp * (fan + 1)

        cacheLabel = torch.zeros(self.batchsize)
        cacheGraph[0] = torch.cat(cacheGraph[0], dim=0)
        cacheGraph[1] = torch.cat(cacheGraph[1], dim=0)
        return cacheGraph, cacheLabel

    def preGraphBatch(self):
        preBatchTime = time.time()
        if self.graphPipe.qsize() >= self.cacheNUM:
            return 0

        nextLoadingTime = time.time()
        if self.trainptr == self.trainLoop:
            logger.debug("触发cache reload ,ptr:{}".format(self.trainptr))
            self.trainptr = 0
            self.initNextGraphData()
        logger.info("next loading cost {:.5f}s".format(time.time() - nextLoadingTime))

        cacheTime = time.time()
        cacheGraph = copy.deepcopy(self.template_cache_graph)
        cacheLabel = copy.deepcopy(self.template_cache_label)
        # TODO:
        sampleIDs = -1 * torch.ones(self.batchsize, dtype=torch.int64)
        logger.info("cache copy graph and label cost {:.5f}s".format(time.time() - cacheTime))

        ##
        createDataTime = time.time()
        batchlen = 0
        if self.trainptr < self.trainLoop - 1:
            # 完整batch
            sampleIDs = self.trainNodes[self.trainptr * self.batchsize:(self.trainptr + 1) * self.batchsize]
            batchlen = self.batchsize
            cacheLabel = self.nodeLabels[sampleIDs]
        else:
            offset = self.trainptr * self.batchsize
            sampleIDs[:self.subGtrainNodesNUM - offset] = self.trainNodes[offset:self.subGtrainNodesNUM]
            batchlen = self.subGtrainNodesNUM - offset
            cacheLabel = self.nodeLabels[sampleIDs[0:self.subGtrainNodesNUM - offset]]
        logger.info("create Data Time cost {:.5f}s".format(time.time() - createDataTime))
        ##

        ##
        sampleTime = time.time()
        blocks, uniqueList = self.sampleNeigGPU_NC(sampleIDs, cacheGraph, batchlen)
        #logger.debug("cacheGraph shape:{}, first graph shape:{}".format(len(cacheGraph),len(cacheGraph[0][0])))
        logger.info("sample subG all cost {:.5f}s".format(time.time() - sampleTime))
        ##

        ##
        featTime = time.time()
        cacheFeat = self.featMerge(uniqueList)
        #cacheFeat=[]
        ##

        ##
        putinTime = time.time()
        cacheData = [blocks, cacheFeat, cacheLabel, batchlen]

        # if self.train_name == "LP":
        #     cacheData = [[], [], [], [], [], []]
        # else:
        #     cacheData = [[], [], [], []]
        self.graphPipe.put(cacheData)
        logger.info("putin pipe time {:.5f}s".format(time.time() - putinTime))
        ##

        self.trainptr += 1
        logger.info("pre graph sample cost {:.5f}s".format(time.time() - preBatchTime))
        logger.info("===============================================")
        return 0

    ########################## 特征提取 ##########################
    def loadingMemFeat(self, rank, preFetch=False,preFeat=None):
        # TODO:如果preFetch为True,则直接从指定位置加载
        print("preFetch=", preFetch)
        if preFetch:
            print("----------------->self.feats :", self.feats)
        filePath = self.dataPath + "/part" + str(rank)
        if preFetch == False:
            tmp_feat = np.fromfile(filePath + "/feat.bin", dtype=np.float32)

        if self.feats == []:
            self.feats = torch.from_numpy(tmp_feat).reshape(-1, self.featlen).to("cuda:0")
        else:
            if preFetch == False:
                tmp_feat = torch.from_numpy(tmp_feat).reshape(-1, self.featlen).to("cuda:0")
                self.feats = torch.cat([self.feats, tmp_feat])
            else:
                print("self.prefeat :", self.prefeat)
                print("self.feats :", self.feats)
                preFeat = torch.from_numpy(preFeat).reshape(-1, self.featlen).to("cuda:0")
                self.feats = torch.cat([self.feats, preFeat])

    def featMerge(self, uniqueList):
        featTime = time.time()
        test = self.feats[uniqueList.to(torch.int64).to(self.feats.device)]
        logger.info("feat merge {}s".format(time.time() - featTime))
        return test

    ########################## 数据调整 ##########################
    def cleanLastModeData(self, mode):
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

    def loadModeData(self, mode):
        logger.info("loading mode:'{}' data".format(mode))
        if "train" == mode:
            self.trainNodeDict, self.trainNodeNumbers = self.loadingTrainID()  # 训练节点字典，训练节点数目
            self.NodeLen = self.trainNUM
        elif "val" == mode:
            self.valNodeDict, self.valNodeNumbers = self.loadingValID()  # 训练节点字典，训练节点数目
            self.NodeLen = self.valNUM
        elif "test" == mode:
            self.testNodeDict, self.testNodeNumbers = self.loadingTestID()  # 训练节点字典，训练节点数目
            self.NodeLen = self.testNUM

    def checkMode(self):
        print("now dataset mode is {}".format(self.mode))

    def create_dgl_block(self, data, num_src_nodes, num_dst_nodes):
        row, col = data
        gidx = dgl.heterograph_index.create_unitgraph_from_coo(2, num_src_nodes, num_dst_nodes, row, col, 'coo')
        g = DGLBlock(gidx, (['_N'], ['_N']), ['_E'])
        return g


def collate_fn(data):
    return data[0]


if __name__ == "__main__":
    dataset = CustomDataset("../../config/dgl_products_graphsage.json",pre_fetch=True)
    with open("../../config/dgl_products_graphsage.json", 'r') as f:
        config = json.load(f)
        batchsize = config['batchsize']
        epoch = config['epoch']
    train_loader = DataLoader(dataset=dataset, batch_size=batchsize, collate_fn=collate_fn)  # pin_memory=True)
    count = 0
    for index in range(3):
        start = time.time()
        loopTime = time.time()
        for graph, feat, label, number in train_loader:
            # src_cat,dst_cat,
            # print(graph)
            # print(src_cat)
            # print(dst_cat)
            # exit()
            count = count + 1
            if count % 20 == 0:
                print("loop time:{:.5f}".format(time.time() - loopTime))
            loopTime = time.time()
