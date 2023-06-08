import concurrent.futures
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data,trainIDs):
        self.data = data
        self.executor = concurrent.futures.ThreadPoolExecutor()
        self.sample = None
        self.trainIDs = trainIDs
        self.tmpFeat = {}
        self.ptrpalce = 0


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        if self.sample is None:
            # 提交数据读取任务给线程池
            future = self.executor.submit(self.read_sample, index)
            self.sample = future
            
        if self.sample.done():
            # 获取任务的返回结果
            result = self.sample.result()
            self.sample = self.executor.submit(self.read_sample, index)
            
            # 在此对上一个任务的结果result进行处理
            
            return result
    
    def prefeat():
        pass

    def read_sample(self, index):

        return self.data[index]
    
def collate_fn(data):
    return data


if __name__ == "__main__":
    data = [i for i in range(100)]
    dataset = CustomDataset(data)