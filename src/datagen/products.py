import torch
from dgl.data import AsNodePredDataset
from ogb.nodeproppred import DglNodePropPredDataset
import numpy as np
import random
import pickle
import dgl

root="/home/bear/workspace/singleGNN/data/dataset"
dataset = AsNodePredDataset(DglNodePropPredDataset('ogbn-products',root=root))
g = dataset[0]
src = g.edges()[0].numpy()
dst = g.edges()[1].numpy()
src.tofile("/raid/bear/products_bin/srcList.bin")
dst.tofile("/raid/bear/products_bin/dstList.bin")
feat = g.ndata['feat'].numpy().tofile("/raid/bear/products_bin/feat.bin")
label = g.ndata['label'].numpy().tofile("/raid/bear/products_bin/label.bin")
torch.nonzero(g.ndata['train_mask']).squeeze().numpy().tofile("/raid/bear/products_bin/trainIDs.bin")
torch.nonzero(g.ndata['val_mask']).squeeze().numpy().tofile("/raid/bear/products_bin/valIDs.bin")
torch.nonzero(g.ndata['test_mask']).squeeze().numpy().tofile("/raid/bear/products_bin/testIDs.bin")


def lp_data_gen(folder_path):
    edgeNUM = 1000000
    neg_num = 1000
    sampled_edge_ids = random.sample(range(g.num_edges()), edgeNUM)
    trainId=sampled_edge_ids[:int(edgeNUM * 0.3)]
    TestId=sampled_edge_ids[int(edgeNUM * 0.3):int(edgeNUM * 0.9)]
    ValId=sampled_edge_ids[int(edgeNUM * 0.9):]


    neg_train_sampler = dgl.dataloading.negative_sampler.Uniform(1)
    train_src, train_neg_dst = neg_train_sampler(g, torch.Tensor(trainId).to(torch.int64))
    raw_train_src=torch.Tensor(g.edges()[0][trainId])
    raw_train_dst=torch.Tensor(g.edges()[1][trainId])


    neg_val_sampler = dgl.dataloading.negative_sampler.Uniform(neg_num)
    val_src,val_neg_dst = neg_val_sampler(g, torch.Tensor(ValId).to(torch.int64))
    raw_val_src=torch.Tensor(g.edges()[0][ValId])
    raw_val_dst=torch.Tensor(g.edges()[1][ValId])
    val_neg_dst.reshape(-1,neg_num)

    neg_test_sampler = dgl.dataloading.negative_sampler.Uniform(neg_num)
    test_src, test_neg_dst = neg_test_sampler(g, torch.Tensor(TestId).to(torch.int64))
    raw_test_src=torch.Tensor(g.edges()[0][TestId])
    raw_test_dst=torch.Tensor(g.edges()[1][TestId])
    test_neg_dst.reshape(-1,neg_num)

    split_pt = {}
    split_pt['train']={}
    split_pt['train']['source_node']=raw_train_src
    split_pt['train']['target_node']=raw_train_dst
    split_pt['train']['target_node_neg']=raw_train_src

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