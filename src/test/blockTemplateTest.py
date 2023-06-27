# 多维转换
import torch as th
from pympler import asizeof
import dgl
import sys
import copy
"""
测试常规采样结果转换为dgl.block训练集
采样结果数据结构
[[src2,dst2],[[src1,dst1]]]
其中src2 = src1 + sampled
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
index 为1的节点:
1 
fan1*1-2*fan1
fan1*fan2*1-2*fan2*fan1 :ID序列

"""
# print(data)
# 给出fanout，batchsize的情况下，填图block
# graphNodes = batchsize

def template_graph_gen():
    fanout = [4,4]
    template = []
    blocks = []
    batchsize = 4

    ptr = 0
    seeds = [i for i in range(1,4)]
    dst = [i for i in range(1,4)]
    src = [i for i in range(1,4)]

    for number in fanout:
        dst = copy.deepcopy(seeds)
        src = copy.deepcopy(seeds)
        ptr = len(src) + 1    
        for ids in seeds:
            for i in range(number-1):
                dst.append(ids)
                src.append(ptr)
                ptr += 1
        template.insert(0,[th.tensor(src),th.tensor(dst)])
        seeds = copy.deepcopy(src)
    # print("="*15+"\n" +"graph template: {}\n".format(template)+"="*15)
    return template

template = template_graph_gen()
for index,mask in enumerate(masks):
    src,dst = template[index]
    src *= mask
    dst *= mask
print(template)

blocks = []
for src,dst in template:
    block = dgl.graph((src, dst))
    block = dgl.to_block(block)
    blocks.append(block)
print(blocks[0].srcdata[dgl.NID])
print(blocks)