import dgl
import torch
import numpy as np
import time
from scipy.sparse import csr_matrix

graphbin=""
edges = np.fromfile(graphbin,dtype=np.int32)
src = torch.tensor(edges[::2]).to(torch.int64)
dst = torch.tensor(edges[1::2]).to(torch.int64)
# g = dgl.graph((src, dst))
# g = g.formats('csc')
# indptr, indices, _ = g.adj_sparse(fmt='csc')

row  = np.array([0, 1, 2, 2, 2, 0])
col  = np.array([0, 2, 2, 0, 1, 2])
data = np.array([1, 2, 3, 4, 5, 6])
m = csr_matrix((data, (row, col)))
indptr = m.indptr
indices = m.indices