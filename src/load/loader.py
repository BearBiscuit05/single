import concurrent.futures
from torch.utils.data import Dataset, DataLoader
from queue import Queue

class CustomDataset(Dataset):
    def __init__(self,data,cacheNUM,batchsize,trainIDs):
        self.data = data
        self.trainIDs = trainIDs
        self.src = [i for i in range(10000)]
        self.bound = [i*100 for i in range(100)]
        self.executor = concurrent.futures.ThreadPoolExecutor(1)
        self.sample_flag = None
        self.batchsize = batchsize
        self.cacheNUM = cacheNUM

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


        if index % self.batchsize == 0 and index+self.batchsize < len(self.data):        
            # 表明一个batch开始预取，则需要重新获得新数据
            self.place += 1
            datalen = min(index+self.batchsize*2,len(self.data))
            preindex = [index+self.batchsize,datalen]
            if self.sample_flag is None:
                future = self.executor.submit(self.read_sample, preindex,self.place%2)
                self.sample_flag = future
            else:
                data = self.sample_flag.result()
                #if self.sample_flag.done():
                self.sample_flag = self.executor.submit(self.read_sample, preindex,self.place%2)     
        elif index % self.batchsize == 0 and index+self.batchsize >= len(self.data):  # last
            data = self.sample_flag.result()
            self.place += 1
        return self.loadingData[(self.place-1)%2][index % self.batchsize]
    
    def prefeat():
        pass

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
    dataset = CustomDataset(data,4,4,data)
    train_loader = DataLoader(dataset=dataset, batch_size=4, collate_fn=collate_fn,pin_memory=True)
    for i in train_loader:
        print(i)