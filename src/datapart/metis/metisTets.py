import dgl
import torch as th
import numpy as np
import argparse
import time
import sys
import os
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset

ORKPATH = "/raid/bear/tmp/com_or.bin"
ITPATH = "/raid/bear/bigdata/it2004.bin"

for PATH in [ITPATH]:
    edges = np.fromfile(PATH,dtype=np.int32)
    odd_indices = edges[1::2]  # 奇数索引（从索引1开始，步长为2）
    even_indices = edges[::2]   # 偶数索引（从索引0开始，步长为2）
    g=dgl.graph((odd_indices,even_indices))
    partList = [4,8,16,32]
    for part in partList:
        dgl.distributed.partition_graph(g, 'test', part, num_hops=1, part_method='metis',out_path='output/')