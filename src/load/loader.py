import concurrent.futures
from torch.utils.data import Dataset, DataLoader
from queue import Queue
import numpy as np
import json


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
        self.readConfig(confPath)
        self.cacheData = []     # 子图存储部分
        self.pipe = Queue()     # 采样存储部分
        self.trainNUM = 0       # 训练集数目
        self.graphTrack = self.randomTrainList() # 训练轨迹
        self.trainNodeDict = self.loadingTrainID() # 训练节点
        self.executor = concurrent.futures.ThreadPoolExecutor(1) # 线程池
        self.trainTrack = self.randomTrainList() # 获得随机序列
        self.sample_flag = None
        
        self.trained = 0
        self.read = 0
        self.loop = ((self.trainNUM-1) // self.batchsize) + 1
        self.read_called = 0

        self.initGraphData()
        print(self.cacheData)
    def __len__(self):  
        return self.trainNUM
    
    def __getitem__(self, index):
        # 数据流驱动函数
        if index % self.batchsize == 0:
            # 调用预取函数
            if self.read_called < self.loop:
                if self.sample_flag is None:
                    future = self.executor.submit(self.preGraphBatch)
                    self.sample_flag = future
                else:
                    data = self.sample_flag.result()
                    #if self.sample_flag.done():
                    self.sample_flag = self.executor.submit(self.preGraphBatch) 
            # 调用实际数据
            if self.pipe.qsize() > 0:
                self.cacheData = self.pipe.get()
            else: #需要等待
                data = self.read_data.result()
                self.cacheData = self.pipe.get()
        return self.cacheData[index % self.batchsize]

    def initGraphData(self):
        print()
        self.loadingGraph(self.trainTrack[0][0])
        self.loadingGraph(self.trainTrack[0][1])

    def readConfig(self,confPath):
        with open(confPath, 'r') as f:
            config = json.load(f)
        self.dataPath = config['datasetpath']+"/"+config['dataset']
        self.batchsize = config['batchsize']
        self.cacheNUM = config['cacheNUM']
        self.partNUM = config['partNUM']
        self.epoch = config['epoch']
        formatted_data = json.dumps(config, indent=4)
        print(formatted_data)

    def loadingTrainID(self):
        # 加载子图所有训练集
        idDict = {}
        
        for index in range(self.partNUM):
            idDict[index] = [i for i in range(10)]
            self.trainNUM += len(idDict[index])
        return idDict

    def loadingGraph(self,subGID):
        # 读取int数组的二进制存储
        # 需要将边界填充到前面的预存图中
        filePath = self.dataPath + "/part" + str(subGID)
        srcdata = np.fromfile(filePath+"/srcList.bin", dtype=np.int32)
        rangedata = np.fromfile(filePath+"/range.bin", dtype=np.int32)
        # 转换为tensor : tensor_data = torch.from_numpy(data)
        print("subG {} read success".format(subGID))
        self.cacheData.append(srcdata)
        self.cacheData.append(rangedata)

    def moveGraph(self):
        self.cacheData[0] = self.cacheData[2]
        self.cacheData[1] = self.cacheData[3]
        # del self.cacheData[3]
        # del self.cacheData[2]
        self.cacheData = self.cacheData[0:2]
       
    def readNeig(self,nodeID):
        return self.src[self.bound[nodeID*2]:self.bound[nodeID*2+1]]

    def prefeat(self):
        pass
    
    def randomTrainList(self):
        epochList = []
        for i in range(self.epoch):
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
        if self.read_called > self.loop:
            return 0
        self.read_called += 1
        print("预取数据部分:{}:{}...".format(self.read,self.read+self.batchsize))
        cacheData = []
        for i in range(self.batchsize):
            sampleID = self.trainIDs[self.read+i]
            sampleG = self.src[self.bound[sampleID]:self.bound[sampleID+1]]
            cacheData.append(sampleG)
        self.pipe.put(cacheData)
        self.read += self.batchsize
        return 0
    
def collate_fn(data):
    return data


if __name__ == "__main__":
    data = [i for i in range(20)]
    dataset = CustomDataset("./config.json")

    # train_loader = DataLoader(dataset=dataset, batch_size=4, collate_fn=collate_fn,pin_memory=True)
    # for i in train_loader:
    #     print(i)