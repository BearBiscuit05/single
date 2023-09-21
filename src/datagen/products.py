"""
Partial code reference: https://github.com/SJTU-IPADS/gnnlab
"""

import torch
from dgl.data import AsNodePredDataset
from ogb.nodeproppred import DglNodePropPredDataset
import numpy as np
import random
import pickle
import dgl
import os

DATA_PATH = '/raid/bear/sgnn'

DOWNLOAD_URL = 'http://snap.stanford.edu/ogb/data/nodeproppred/products.zip'
RAW_DATA_DIR = DATA_PATH +'/raw_dataset'
PAPERS_RAW_DATA_DIR = f'{RAW_DATA_DIR}/products'
OUTPUT_DATA_DIR = DATA_PATH + '/dataset/products'


def download_data():
    print('Download data...')
    if not os.path.exists(f'{RAW_DATA_DIR}/products.zip'):
        assert(os.system(
            f'wget {DOWNLOAD_URL} -O {RAW_DATA_DIR}/products.zip') == 0)
    else:
        print('Already downloaded.')

    print('Unzip data...')
    if not os.path.exists(f'{PAPERS_RAW_DATA_DIR}/unzipped'):
        assert(os.system(
            f'cd {RAW_DATA_DIR}; unzip {RAW_DATA_DIR}/products.zip') == 0)
        assert(os.system(f'touch {PAPERS_RAW_DATA_DIR}/unzipped') == 0)
    else:
        print('Already unzipped...')

def write_meta():
    print('Writing meta file...')
    with open(f'{OUTPUT_DATA_DIR}/meta.txt', 'w') as f:
        f.write('{}\t{}\n'.format('NUM_NODE', 123718280))
        f.write('{}\t{}\n'.format('NUM_EDGE', 2449029))
        f.write('{}\t{}\n'.format('FEAT_DIM', 100))
        f.write('{}\t{}\n'.format('NUM_CLASS', 47))
        f.write('{}\t{}\n'.format('NUM_TRAIN_SET', 196615))
        f.write('{}\t{}\n'.format('NUM_VALID_SET', 39323))
        f.write('{}\t{}\n'.format('NUM_TEST_SET', 2213091))

def convert_data_tobin():
    dataset = AsNodePredDataset(DglNodePropPredDataset('ogbn-products',root=RAW_DATA_DIR))
    g = dataset[0]
    src = g.edges()[0].numpy()
    dst = g.edges()[1].numpy()
    src.tofile(OUTPUT_DATA_DIR + "/srcList.bin")
    dst.tofile(OUTPUT_DATA_DIR + "/dstList.bin")
    g.ndata['feat'].numpy().tofile(OUTPUT_DATA_DIR + "/feats.bin")
    g.ndata['label'].numpy().tofile(OUTPUT_DATA_DIR + "/labels.bin")
    torch.nonzero(g.ndata['train_mask']).squeeze().numpy().tofile(OUTPUT_DATA_DIR + "/trainIDs.bin")
    torch.nonzero(g.ndata['val_mask']).squeeze().numpy().tofile(OUTPUT_DATA_DIR + "/valIDs.bin")
    torch.nonzero(g.ndata['test_mask']).squeeze().numpy().tofile(OUTPUT_DATA_DIR + "/testIDs.bin")


def lp_data_gen(g,edgeNUM,neg_num):
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
    split_pt['train']['target_node_neg']=train_neg_dst

    split_pt['valid']={}
    split_pt['valid']['source_node']=raw_val_src
    split_pt['valid']['target_node']=raw_val_dst
    split_pt['valid']['target_node_neg']=val_neg_dst.reshape(-1,neg_num)

    split_pt['test']={}
    split_pt['test']['source_node']=raw_test_src
    split_pt['test']['target_node']=raw_test_dst
    split_pt['test']['target_node_neg']=test_neg_dst.reshape(-1,neg_num)

    file_name = OUTPUT_DATA_DIR+"/split_lp_dict.pkl"
    pick_file = open(file_name,'wb')
    pickle.dump(split_pt,pick_file)
    pick_file.close()

if __name__ == '__main__':
    pass