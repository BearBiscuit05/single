import time
import copy
import torch
import dgl

dst = [0,0,0,1,1,1,2,2,2]
src = [4,5,6,7,8,9,10,11,12]
g = dgl.graph((src,dst))
frontier = g.sample_neighbors([1], -1)
print(frontier.edges())