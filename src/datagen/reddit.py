import torch
import dgl
import numpy as np
import random
import pickle
from dgl.data import RedditDataset

def load_reddit(self_loop=True):
    data = RedditDataset(self_loop=self_loop,raw_dir='./data/dataset/')
    g = data[0]
    g.ndata['feat'] = g.ndata.pop('feat')
    g.ndata['label'] = g.ndata.pop('label')
    train_idx = []
    val_idx = []
    test_idx = []
    for index in range(len(g.ndata['train_mask'])):
        if g.ndata['train_mask'][index] == 1:
            train_idx.append(index)
    for index in range(len(g.ndata['val_mask'])):
        if g.ndata['val_mask'][index] == 1:
            val_idx.append(index)
    for index in range(len(g.ndata['test_mask'])):
        if g.ndata['test_mask'][index] == 1:
            test_idx.append(index)
    return g, data,train_idx,val_idx,test_idx

g, dataset,train_idx,val_idx,test_idx= load_reddit()

src = g.edges()[0].numpy()
dst = g.edges()[1].numpy()
src.tofile("/raid/bear/reddit_bin/srcList.bin")
dst.tofile("/raid/bear/reddit_bin/dstList.bin")

feat = g.ndata['feat'].numpy().tofile("/raid/bear/reddit_bin/feat.bin")
label = g.ndata['label'].numpy().tofile("/raid/bear/reddit_bin/label.bin")

torch.Tensor(train_idx).to(torch.int64).numpy().tofile("/raid/bear/reddit_bin/trainIDs.bin")
torch.Tensor(val_idx).to(torch.int64).numpy().tofile("/raid/bear/reddit_bin/valIDs.bin")
torch.Tensor(test_idx).to(torch.int64).numpy().tofile("/raid/bear/reddit_bin/testIDs.bin")

def gen_lp():
    g = dataset[0]
    g.ndata['feat'] = g.ndata.pop('feat')
    g.ndata['label'] = g.ndata.pop('label')

    folder_path="/home/wsy/single-gnn/data/dataset/reddit_69f818f5/split_lp"
    print("reddit edges:",g.num_edges())
    edgeNUM = 3000000
    neg_num = 3000
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
