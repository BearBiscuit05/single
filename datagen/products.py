import torch
from dgl.data import AsNodePredDataset
from ogb.nodeproppred import DglNodePropPredDataset
import numpy as np

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