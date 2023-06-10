import concurrent.futures
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, data,trainIDs):
        self.data = data
        self.executor = concurrent.futures.ThreadPoolExecutor()
        self.sample = None
        self.trainIDs = trainIDs
        self.ptrpalce = 0
        self.batchsize = 4
        # tmp = [i for i in range(self.batchsize)]
        self.loadingData = [[i for i in range(self.batchsize)],[i for i in range(self.batchsize)]]
        self.place = 0
        self.read_sample([0,self.batchsize], self.place)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        if index % self.batchsize == 0 and index+self.batchsize < len(self.data):        
            self.place += 1
            datalen = min(index+self.batchsize*2,len(self.data))
            preindex = [index+self.batchsize,datalen]
            if self.sample is None:
                future = self.executor.submit(self.read_sample, preindex,self.place%2)
                self.sample = future
            else:
                data = self.sample.result()
                #if self.sample.done():
                self.sample = self.executor.submit(self.read_sample, preindex,self.place%2)     
        elif index % self.batchsize == 0 and index+self.batchsize >= len(self.data):  # last
            data = self.sample.result()
            self.place += 1
        return self.loadingData[(self.place-1)%2][index % self.batchsize]
    
    def prefeat():
        pass

    def read_sample(self, index, place):
        print("预取数据部分:{}:{}...".format(index[0],index[1]))
        for i in range(index[0],index[1]):
            self.loadingData[place][i%self.batchsize] = self.data[i] * 2
        return 0
    
def collate_fn(data):
    return data


if __name__ == "__main__":
    data = [i for i in range(20)]
    dataset = CustomDataset(data,1)
    train_loader = DataLoader(dataset=dataset, batch_size=4, collate_fn=collate_fn,pin_memory=True)
    for i in train_loader:
        print(i)