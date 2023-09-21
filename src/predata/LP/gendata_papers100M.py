import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
import dgl
import dgl.nn as dglnn
from dgl.data import AsNodePredDataset
from dgl.dataloading import DataLoader, NeighborSampler, MultiLayerFullNeighborSampler
from ogb.nodeproppred import DglNodePropPredDataset
import os
import numpy as np
import random
import pickle
import copy

from dgl.dataloading import (
	as_edge_prediction_sampler,
	DataLoader,
	MultiLayerFullNeighborSampler,
	negative_sampler,
	NeighborSampler,
)

if __name__ == '__main__':
	dataset = AsNodePredDataset(DglNodePropPredDataset('ogbn-papers100M',root="/home/bear/workspace/singleGNN/data/dataset"))
	folder_path="/home/wsy/single-gnn/data/dataset/ogbn_papers100M/split_lp"
	g = dataset[0]
	# print("papers100M edges:",g.num_edges())
	# edgeNUM = 1615685872
	# edgeNUM = 1600000000
	edgeNUM = 1000000
	neg_num = 1000
	sampled_edge_ids = torch.randperm(g.num_edges()).to(torch.int64)[:edgeNUM]
	trainId=sampled_edge_ids[:int(edgeNUM * 0.3)]
	TestId=sampled_edge_ids[int(edgeNUM * 0.3):int(edgeNUM * 0.9)]
	ValId=sampled_edge_ids[int(edgeNUM * 0.9):]
	
	neg_train_sampler = dgl.dataloading.negative_sampler.Uniform(1)
	train_src, train_neg_dst = neg_train_sampler(g, trainId)
	raw_train_src=g.edges()[0][trainId]
	raw_train_dst=g.edges()[1][trainId]

	neg_val_sampler = dgl.dataloading.negative_sampler.Uniform(neg_num)
	val_src,val_neg_dst = neg_val_sampler(g,ValId)
	raw_val_src=g.edges()[0][ValId]
	raw_val_dst=g.edges()[1][ValId]
	val_neg_dst.reshape(-1,neg_num)

	neg_test_sampler = dgl.dataloading.negative_sampler.Uniform(neg_num)
	test_src, test_neg_dst = neg_test_sampler(g,TestId)
	raw_test_src=g.edges()[0][TestId]
	raw_test_dst=g.edges()[1][TestId]
	test_neg_dst.reshape(-1,neg_num)

	split_pt = {}
	split_pt['train']={}
	split_pt['train']['source_node']=raw_train_src
	split_pt['train']['target_node']=raw_train_dst
	split_pt['train']['target_node_neg']=train_neg_dst.reshape(-1,neg_num)

	split_pt['valid']={}
	split_pt['valid']['source_node']=raw_val_src
	split_pt['valid']['target_node']=raw_val_dst
	split_pt['valid']['target_node_neg']=val_neg_dst.reshape(-1,neg_num)

	split_pt['test']={}
	split_pt['test']['source_node']=raw_test_src
	split_pt['test']['target_node']=raw_test_dst
	split_pt['test']['target_node_neg']=test_neg_dst.reshape(-1,neg_num)

	file_name = folder_path+"/split_dict.pkl"
	pick_file = open(file_name,'wb')
	pickle.dump(split_pt,pick_file)
	pick_file.close()
	#torch.save(split_lp, file_name)
	#dgl.save_graphs(file_name,newg)