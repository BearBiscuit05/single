import torch
import signn
import numpy as np

filePath = "./../../data/products_4/part0"

srcdata = np.fromfile(filePath+"/srcList.bin", dtype=np.int32)
srcdata = torch.tensor(srcdata,device='cuda:0')
rangedata = np.fromfile(filePath+"/range.bin", dtype=np.int32)
rangedata = torch.tensor(rangedata,device='cuda:0')

batch=512
seeds = []
for i in range(batch):
    seeds.append(i)

# 提供eid
sampleIDs = torch.Tensor(seeds).to(torch.int32).to("cuda:0")
fan_num = 2
seed_num = len(sampleIDs)
length = fan_num * seed_num
out_src = torch.zeros(length).to(torch.int32).to("cuda:0")
out_dst = torch.zeros(length).to(torch.int32).to("cuda:0")
out_num = torch.Tensor([0]).to(torch.int64).to("cuda:0")
signn.torch_sample_hop(
    srcdata,rangedata,
    sampleIDs,seed_num,fan_num,
    out_src,out_dst,out_num)

# 获得src,dst
out_src = out_src[:out_num.item()]
out_dst = out_dst[:out_num.item()]

# 获得负采样
neg_dst = torch.randint(low=0, high=len(rangedata)//2, size=out_src.shape).to(torch.int32).to("cuda:0")
all_tensor = torch.cat([out_src,out_dst,neg_dst])
src_cat = torch.cat([out_src,out_src])
dst_cat = torch.cat([out_dst,new_random_tensor])

raw_src = copy.deepcopy(out_src)
raw_dst = copy.deepcopy(out_dst)
edgeNUM = len(src_cat)      
uniqueNUM = torch.Tensor([0]).to(torch.int64).to('cuda:0')
unique = torch.zeros(len(all_tensor),dtype=torch.int32).to('cuda:0')

signn.torch_graph_mapping(all_tensor,src_cat,dst_cat,src_cat,dst_cat,unique,edgeNUM,uniqueNUM)
unique = unique[:uniqueNUM.item()]

# unique 为采样集
fanout = [10,10,5]
sampleIDs = copy.deepcopy(unique)
seed_num = len(sampleIDs)

for fan_num in fanout:
    length = seed_num * fan_num
    out_src = torch.zeros(length).to(torch.int32).to("cuda:0")
    out_dst = torch.zeros(length).to(torch.int32).to("cuda:0")
    out_num = torch.Tensor([0]).to(torch.int64).to("cuda:0")
    signn.torch_sample_hop(
        srcdata,rangedata,
        sampleIDs,seed_num,fan_num,
        out_src,out_dst,out_num)
    indices = torch.where((out_src.unsqueeze(1) == raw_src) & (out_dst.unsqueeze(1) == raw_dst))
    edge_indices = indices[0]
    out_src[] = -1
    out_dst[] = -1
 