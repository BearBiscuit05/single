# Multidimensional transformation
import torch as th
from pympler import asizeof
import dgl
import sys
import copy
"""
Test the conversion of conventional sampling results to a dgl.block training set
Sample result data structure
[[src2,dst2],[[src1,dst1]]]
src2 = src1 + sampled
"""

fanout = [4,4]

dst1 = [10, 11, 12, 10, 10, -1, 11, -1, -1, 12, 12, 12]
src1 = [10, 11, 12, 23, 24, -1, 26, -1, -1, 29, 30, 31]
dst2 = [10, 11, 12, 23, 24, -1, 26, -1, -1, 29, 30, 31]
src2 = [10, 11, 12, 23, 24, -1, 26, -1, -1, 29, 30, 31]
zero_info = {}
zero_info[0] = []
zero_info[1] = []
id = 40
for info in src1:
    for i in range(3):
        if info == -1:
            dst2.append(-1)
            src2.append(-1)
        else:
            dst2.append(info)
            src2.append(id)
            id += 1
graph = [] 
graph.append([th.tensor(src2),th.tensor(dst2)])
graph.append([th.tensor(src1),th.tensor(dst1)])
# print(graph)
masks = []
for src, dst in graph:
    layer_mask = th.ge(src, 0)
    masks.append(layer_mask)
# print(masks)
""" 
node whose index is 1:
1 
fan1*1-2*fan1
fan1*fan2*1-2*fan2*fan1 : ID sequence

"""
# print(data)
# when fanout，batchsize are given，get graph block
# graphNodes = batchsize

def genBlockTemplate():
    template = []
    blocks = []
    ptr = 0
    fanout=[4,4]
    batchsize=4
    seeds = [i for i in range(1,batchsize+1)]
    for number in fanout:
        dst = copy.deepcopy(seeds)
        src = copy.deepcopy(seeds)
        ptr = len(src) + 1    
        for ids in seeds:
            for i in range(number-1):
                dst.append(ids)
                src.append(ptr)
                ptr += 1
        seeds = copy.deepcopy(src)
        src.append(0)
        dst.append(0)
        template.insert(0,[th.tensor(src),th.tensor(dst)])
    return template

template = genBlockTemplate()
print(template)
for index,mask in enumerate(masks):
    src,dst = template[index]
    src *= mask
    dst *= mask
# print(template)

blocks = []
for src,dst in template:
    block = dgl.graph((src, dst))
    block = dgl.to_block(block)
    blocks.append(block)
print(blocks[0].srcdata[dgl.NID])
print(blocks)