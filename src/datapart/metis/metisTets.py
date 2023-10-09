import dgl
import torch as th
import numpy as np
import argparse
import time
import sys
import os
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset

edges = np.loadtxt("/raid/bear/tmp/tw-2010.txt",delimiter='\t',dtype=np.int32)
edges = edges.reshape(1,-1)[0]
odd_indices = edges[1::2]  # 奇数索引（从索引1开始，步长为2）
even_indices = edges[::2]   # 偶数索引（从索引0开始，步长为2）
g=dgl.graph((odd_indices,even_indices))
partList = [4,8,128,256]
for part in partList:
    dgl.distributed.partition_graph(g, 'test', part, num_hops=1, part_method='metis',out_path='output/')