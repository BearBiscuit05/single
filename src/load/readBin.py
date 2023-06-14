import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self):
        self.tensor1 = torch.zeros(50)
        self.tensor2 = torch.zeros(100)
        self.tensor3 = torch.zeros(50)
        self.tensor4 = torch.zeros(100)
        self.l = []


    def change(self):
        self.l[0] += 1

    def add(self):
        t1,t2 = self.newdata()
        self.l.append(t1)
        self.l.append(t2)
        

    def remove(self):
        self.tensor3 = None
        self.tensor4 = None 

    def print(self):
        print(self.l)

    def newdata(self):
        tmp1 = torch.zeros(50)
        tmp2 = torch.zeros(100)
        return tmp1,tmp2

if __name__ == "__main__":
    my_list = [1, 2, 3, 4, 5]

    # 获取索引为 1 的元素的引用
    tmp = my_list[1]

    # 修改 tmp
    tmp = 10

    print(my_list)  # 输出: [1, 10, 3, 4, 5]
    print(tmp)  # 输出: 10

