import concurrent.futures
from torch.utils.data import Dataset, DataLoader
from queue import Queue
import numpy as np
import json
import time
import mmap
from dgl.heterograph import DGLBlock
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
        # 获得训练基本信息
        self.cacheData = []     # 子图存储部分
        self.graphPipe = Queue()    # 采样存储管道
        self.blockPipe = Queue()    # 采样图+特征
        self.sampledSubG = []   # 采样子图存储位置
        self.sampledfeat = []   # 采样子图特征存储
        self.trainNUM = 0       # 训练集总数目
        
        # config json 部分
        self.dataPath = ''
        self.batchsize = 0
        self.cacheNUM = 0
        self.partNUM = 0
        self.epoch = 0
        self.preRating = 0
        self.featlen = 0
        self.idbound = []
        # ================
        
        self.trained = 0
        self.trainptr = 0   # 当前训练集读取位置
        self.loop = 0
        self.subGptr = -1   # 子图训练指针，记录当前已经预取的位置
        self.batch_called = 0   # 批预取函数调用次数
        self.trainingGID = 0 # 当前训练子图的ID
        self.nextGID = 0     # 下一个训练子图
        self.trainNodes = []            # 子图训练节点记录   
        self.graphNodeNUM = 0 # 当前训练子图节点数目
        self.graphEdgeNUM = 0 # 当前训练子图边数目
        self.subGtrainNodesNUM = 0 # 当前训练子图训练节点
        self.readConfig(confPath)
        self.trainNodeDict = self.loadingTrainID() # 训练节点
        self.executor = concurrent.futures.ThreadPoolExecutor(1) # 线程池
        self.trainSubGTrack = self.randomTrainList() # 训练轨迹
        
        # 特征部分
        self.readfile = []  # 包含两个句柄/可能有三个句柄
        self.mmapfile = []  
        self.loadingFeatFileHead()      # 读取特征文件

        # 数据预取
        self.loadingGraph()
        self.initNextGraphData()
        self.sample_flag = self.executor.submit(self.preGraphBatch) #发送采样命令
        
    def __len__(self):  
        return self.trainNUM
    
    def __getitem__(self, index):
        # 批数据预取 缓存1个
        if index % self.preRating == 0:
            # 调用预取函数
            # data = self.sample_flag.result()
            self.sample_flag = self.executor.submit(self.preGraphBatch) 
        
        # 获取采样数据
        if index % self.batchsize == 0:
            # 调用实际数据
            if self.graphPipe.qsize() > 0:
                cacheData = self.graphPipe.get()
                self.sampledSubG = cacheData[0]
                self.sampledfeat = cacheData[1]
            else: #需要等待
                data = self.sample_flag.result()
                cacheData = self.graphPipe.get()
                self.sampledSubG = cacheData[0]
                self.sampledfeat = cacheData[1]
        return self.sampledSubG[index % self.batchsize],self.sampledfeat[index % self.batchsize]

    def initNextGraphData(self):
        # 查看是否需要释放
        if len(self.cacheData) > 2:
            self.moveGraph()
        # 对于将要计算的子图(已经加载)，修改相关信息
        self.trainingGID = self.trainSubGTrack[self.subGptr//self.partNUM][self.subGptr%self.partNUM]
        self.trainNodes = self.trainNodeDict[self.trainingGID]
        self.subGtrainNodesNUM = len(self.trainNodes)
        self.loop = ((self.subGtrainNodesNUM - 1) // self.batchsize) + 1
        self.graphNodeNUM = int(len(self.cacheData[1]) / 2 )# 获取当前节点数目
        self.graphEdgeNUM = len(self.cacheData[0])
        
        # 对于辅助计算的子图，进行加载，以及加载融合边
        self.loadingGraph()
        self.nextGID = self.trainSubGTrack[self.subGptr//self.partNUM][self.subGptr%self.partNUM]
        self.loadingHalo()
        print("当前加载图为:{},下一个图:{},图训练集规模:{},图节点数目:{},图边数目:{},循环计算次数{}"\
                        .format(self.trainingGID,self.nextGID,self.subGtrainNodesNUM,\
                        self.graphNodeNUM,self.graphEdgeNUM,self.loop))
        
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
        self.idbound = config['idbound']
        formatted_data = json.dumps(config, indent=4)
        print(formatted_data)

    def loadingTrainID(self):
        # 加载子图所有训练集
        idDict = {}     
        for index in range(self.partNUM):
            idDict[index] = [i for i in range(10)]
            self.trainNUM += len(idDict[index])
        return idDict

    def loadingGraph(self):
        # 读取int数组的二进制存储， 需要将边界填充到前面的预存图中
        # 由self.subGptr变量驱动,读取时是结构+特征
        self.subGptr += 1
        subGID = self.trainSubGTrack[self.subGptr//self.partNUM][self.subGptr%self.partNUM]
        filePath = self.dataPath + "/part" + str(subGID)
        srcdata = np.fromfile(filePath+"/srcList.bin", dtype=np.int32)
        rangedata = np.fromfile(filePath+"/range.bin", dtype=np.int32)
        
        # 转换为tensor : tensor_data = torch.from_numpy(data)
        self.cacheData.append(srcdata)
        self.cacheData.append(rangedata)

    def loadingFeatFileHead(self):
        for index in range(self.partNUM):
            filePath = self.dataPath + "/part" + str(index)
            file = open(filePath+"/feat_"+str(index)+".bin", "r+b")
            self.readfile.append(file)
            self.mmapfile.append(mmap.mmap(self.readfile[-1].fileno(), 0, access=mmap.ACCESS_READ))
        print("mmap file success...")

    def closeMMapFileHead(self):
        for file in self.mmapfile:
            file.close()
        for file in self.readfile:
            file.close()
        
    def moveGraph(self):
        self.cacheData[0] = self.cacheData[2]
        self.cacheData[1] = self.cacheData[3]
        self.cacheData = self.cacheData[0:2]
            
    def readNeig(self,nodeID):
        return self.src[self.bound[nodeID*2]:self.bound[nodeID*2+1]]

    def loadingHalo(self):
        # 要先加载下一个子图，然后再加载halo( 当前<->下一个 )
        # TODO 读取halo
        filePath = self.dataPath + "/part" + str(self.trainingGID)
        # edges 
        edges = np.fromfile(filePath+"/halo"+str(self.nextGID)+".bin", dtype=np.int32)
        lastid = -1
        startidx = -1
        endidx = -1
        nextidx = -1
        srcList = self.cacheData[0]
        bound = self.cacheData[1]
        for index in range(int(len(edges) / 2)):
            src = edges[index*2]
            dst = edges[index*2 + 1]
            if self.nextGID == 0: # 对于下一子图为0，会需要进行修改
                src += self.graphNodeNUM
            if dst != lastid:
                startidx = self.cacheData[1][dst*2]
                endidx = self.cacheData[1][dst*2+1]
                try:
                    next = self.cacheData[1][dst*2+2]
                except:
                    next = self.graphEdgeNUM
                lastid = dst
                if endidx < next:
                    self.cacheData[0][endidx] = src
                    endidx += 1
            else:
                if endidx < next:
                    self.cacheData[0][endidx] = src
                    endidx += 1
          
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
    
    def preGraphBatch(self):
        # 如果当前管道已经被充满，则不采样，该函数直接返回
        if self.graphPipe.qsize() >= self.cacheNUM:
            return 0
        # 在当前cache中进行采样，如果当前cache已经失效，则需要reload新图
        # 常规预取
        self.batch_called += 1
        if self.trainptr + self.batchsize >= self.subGtrainNodesNUM:
            cacheData = []
            print("[more]从图{}预取数据部分:{}:{}...".format(self.trainingGID,self.trainptr,self.subGtrainNodesNUM))
            # 当前采样图的最后一个批            
            # 先采样剩余部分
            for index in range(0,self.subGtrainNodesNUM-self.trainptr):
                sampleID = self.trainNodes[self.trainptr+index]
                sampleG = self.cacheData[0][self.cacheData[1][sampleID*2]:self.cacheData[1][sampleID*2+1]]
                sampleID = [sampleID for i in range(len(sampleG))]
                cacheData.append([sampleG,sampleID])
            # 重加载
            self.batch_called = 0 #重置
            self.trainptr = 0
            self.initNextGraphData() # 当前epoch采样已经完成，则要预取下轮子图数据
            # 补充采样
            left = self.batchsize - len(cacheData)
            print("[more]从图{}预取数据部分:{}:{}...".format(self.trainingGID,0,left))
            for index in range(left):
                sampleID = self.trainNodes[self.trainptr+index]
                sampleG = self.cacheData[0][self.cacheData[1][sampleID*2]:self.cacheData[1][sampleID*2+1]]
                sampleID = [sampleID for i in range(len(sampleG))]
                cacheData.append([sampleG,sampleID])
            cacheData = self.featMerge(cacheData)
            self.graphPipe.put(cacheData)
            self.trainptr = left # 循环读取
            return 0
        else:
            #bound = min(self.trainptr+self.batchsize,self.trainNUM)
            print("从图{}预取数据部分:{}:{}...".format(self.trainingGID,self.trainptr,self.trainptr+self.batchsize))
            cacheData = []
            for i in range(self.batchsize):
                sampleID = self.trainNodes[self.trainptr+i]
                sampleG = self.cacheData[0][self.cacheData[1][sampleID*2]:self.cacheData[1][sampleID*2+1]]
                sampleID = [sampleID for i in range(len(sampleG))]
                cacheData.append([sampleG,sampleID])
            cacheData = self.featMerge(cacheData)
            self.graphPipe.put(cacheData)
            self.trainptr = self.trainptr + self.batchsize # 循环读取
            return 0
    
    def featMerge(self,SubG):
        # 获取采样子图(SubG) 转换为训练子图(block)
        # mmap返回特征
        batchFeats = []
        float_size = np.dtype(float).itemsize
        for sampledG in SubG:
            feats = []
            for nodeID in sampledG[0]:
                if nodeID < self.graphNodeNUM: # 本地抽取
                    feat = np.frombuffer(self.mmapfile[self.trainingGID], dtype=float, offset=nodeID*self.featlen* float_size, count=self.featlen)
                else:
                    if self.nextGID == 0:
                        # graph_0
                        nodeID -= self.graphNodeNUM
                        feat = np.frombuffer(self.mmapfile[self.nextGID], dtype=float, offset=nodeID*self.featlen* float_size, count=self.featlen)
                    else:
                        nodeID -= self.idbound[self.nextGID][0]
                        feat = np.frombuffer(self.mmapfile[self.nextGID], dtype=float, offset=nodeID*self.featlen* float_size, count=self.featlen)
                feats.append(feat)
            batchFeats.append(feats)
        # self.nextGID = 0     # 下一个训练子图
        # 存储到下一个管道
        block = [SubG,batchFeats]
        return block
        
def collate_fn(data):
    return data


if __name__ == "__main__":
    data = [i for i in range(20)]
    dataset = CustomDataset("./config.json")
    with open("./config.json", 'r') as f:
        config = json.load(f)
        batchsize = config['batchsize']
        epoch = config['epoch']
    
    train_loader = DataLoader(dataset=dataset, batch_size=4, collate_fn=collate_fn,pin_memory=True)
    time.sleep(2)
    for index in range(epoch):
        print("="*15,index,"="*15)
        for i in train_loader:
            print(i)
        print("="*15,index,"="*15)