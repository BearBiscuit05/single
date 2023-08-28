import numpy as np
import torch
import time

trainIDs = torch.load(filePath+"/trainID.bin")
trainIDs = trainIDs.to(torch.uint8).nonzero().squeeze()
labels = torch.from_numpy(np.fromfile(filePath+"/label.bin", dtype=np.int64)).to(torch.int64)
tmp = labels[trainIDs.to(torch.bool)]
print(tmp.max(),tmp.min())