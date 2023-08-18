import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
import dgl
import dgl.nn as dglnn
from dgl.data import AsNodePredDataset
from dgl.dataloading import DataLoader, NeighborSampler, MultiLayerFullNeighborSampler
from ogb.nodeproppred import DglNodePropPredDataset
import tqdm
import argparse
import sklearn.metrics
import numpy as np
import time

# srcList.bin
# if __name__ == '__main__':
# 	srcListBinFile1 = "/home/bear/workspace/singleGNN/data/products_4/part0/srcList.bin"
# 	srcListBinData1 = np.fromfile(srcListBinFile1,dtype=np.int32).tolist()
# 	rangeBinFile1 = "/home/bear/workspace/singleGNN/data/products_4/part0/range.bin"
# 	rangeBinData1 = np.fromfile(rangeBinFile1,dtype=np.int32).tolist()
# 	for j in range(len(rangeBinData1)//2):
# 		l,r = rangeBinData1[j*2],rangeBinData1[j*2+1]
# 		if l == r:
# 			print("")
# 			continue
# 		srcids = srcListBinData1[l:r]
# 		print("%d: "%j,end="")
# 		for srcid in srcids:
# 			print(srcid,end=" ")
# 		print("")

# halo
# if __name__ == '__main__':
# 	haloFile = "/home/bear/workspace/singleGNN/data/products_4/part0/halo1.bin"
# 	halo = np.fromfile(haloFile,dtype=np.int32).tolist()
# 	halo_boundFile = "/home/bear/workspace/singleGNN/data/products_4/part0/halo1_bound.bin"
# 	halo_bound = np.fromfile(halo_boundFile,dtype=np.int32).tolist()
# 	for i in range(len(halo_bound)-1):
# 		l,r = halo_bound[i],halo_bound[i+1]
# 		srcids = []
# 		if l == r:
# 			print("")
# 			continue
# 		for j in range(l,r,2):
# 			srcids.append(halo[j])
# 			if halo[j+1] != i:
# 				print("halo_bound error!")
# 		print("%d: "%i,end="")
# 		for srcid in srcids:
# 			print("%d"%srcid,end=" ")
# 		print("")

# trainIDs
# if __name__ == '__main__':
# 	trainIDBinFile = "/home/bear/workspace/singleGNN/data/products_4/part3/trainID.bin"
# 	trainIDBinData = torch.load(trainIDBinFile).to(torch.uint8).nonzero().squeeze()
# 	for trainID in trainIDBinData:
# 		print(trainID.item())

# 全图处理
if __name__ == '__main__':
	dataset = AsNodePredDataset(DglNodePropPredDataset('ogbn-products'))
	graph = dataset[0].to('cuda:0')
	srcids = graph.edges()[0]#.to(torch.int32)#.numpy()
	dstids = graph.edges()[1]#.to(torch.int32)#.numpy()
	print(len(srcids))
	#print(srcids[0:10])
	#print(dstids[0:10])
	# edges = torch.zeros(2*len(srcids),dtype=torch.int32,device='cuda:0')
	# for index in range(len(srcids)):
	# 	edges[2*index] = srcids[index]
	# 	edges[2*index+1] = dstids[index]
	# edges = edges.to('cpu').numpy()
	# edges.tofile('products4_edges.bin')

	# trainIDs = graph.ndata['train_mask'].nonzero().squeeze().to(torch.int32).to('cpu').numpy()
	# trainIDs.tofile("ogb-products-trainID.bin")