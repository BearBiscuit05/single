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
    dataset = CustomDataset()
    dataset.add()
    dataset.print()
    dataset.add()
    dataset.print()
    dataset.change()
    dataset.print()