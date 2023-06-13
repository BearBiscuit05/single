import concurrent.futures
from torch.utils.data import Dataset, DataLoader
from queue import Queue
import numpy as np
import json
class CustomDataset(Dataset):
    def __init__(self,confPath):
        with open(confPath, 'r') as f:
            config = json.load(f)
        self.dataPath = config['datasetpath']+"/"+config['dataset']
        self.batchsize = config['batchsize']
        self.cacheNUM = config['cacheNUM']
        self.partNUM = config['partNUM']
        self.epoch = config['epoch']

        self.trainTrack = self.randomTrainList()
        self.src,self.bound,self.trainIDs = self.loadingGraph(self.dataPath+"/part0")
        self.executor = concurrent.futures.ThreadPoolExecutor(1)
        self.sample_flag = None
        
        self.cacheData = []
        self.pipe = Queue()
        self.trained = 0
        self.read = 0
        self.loop = ((len(self.trainIDs)-1) // self.batchsize) + 1
        self.read_called = 0
        for i in range(2):
            self.preGraphBatch()

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
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

    def loadingGraph(self,filePath):
        # 读取int数组的二进制存储
        srcdata = np.fromfile(filePath+"/srcList.bin", dtype=np.int32)
        rangedata = np.fromfile(filePath+"/range.bin", dtype=np.int32)
        #trainIds = np.fromfile(filePath+"/trainIds.bin", dtype=np.int32)
        trainIds = [i for i in range(10)]
        # 转换为tensor : tensor_data = torch.from_numpy(data)
        print(srcdata,rangedata)
        return srcdata,rangedata,trainIds

    def readNeig(self,nodeID):
        return self.src[self.bound[nodeID*2]:self.bound[nodeID*2+1]]

    def prefeat(self):
        pass
    
    def randomTrainList(self):
        epochList = []
        for i in range(self.epoch):
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
    dataset = CustomDataset("./processed/part0",4,4,data)

    # train_loader = DataLoader(dataset=dataset, batch_size=4, collate_fn=collate_fn,pin_memory=True)
    # for i in train_loader:
    #     print(i)