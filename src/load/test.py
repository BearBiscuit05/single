import torch
import signn
import numpy as np

for i in range(100000):
    filePath="./../../data/reddit_8/part1"
    srcdata = np.fromfile(filePath+"/srcList.bin", dtype=np.int32)
    srcdata = torch.tensor(srcdata,device=('cuda:0'))
    rangedata = np.fromfile(filePath+"/range.bin", dtype=np.int32)
    rangedata = torch.tensor(rangedata,device=('cuda:0'))
    seeds = torch.load('sids.pt')
    seed_num = len(seeds)
    # print("seed num :",seed_num)
    # print("seed :",seeds.shape)
    tmp = seed_num
    fan = 25
    cacheGraph = [[],[]]
    dst = torch.full((tmp * fan,), -1, dtype=torch.int32).to("cuda:0")  # 使用PyTorch张量，指定dtype
    src = torch.full((tmp * fan,), -1, dtype=torch.int32).to("cuda:0")  # 使用PyTorch张量，指定dtype
    out_num = torch.Tensor([0]).to(torch.int64).to('cuda:0')
    signn.torch_sample_hop(
        srcdata,rangedata,
        seeds,seed_num,fan,
        src,dst,out_num)