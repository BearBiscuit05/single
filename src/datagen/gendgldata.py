import numpy as np
import dgl

TWBINPATH="/raid/bear/dataset/twitter/twitter-2010.bin"
TWDGLPATH="/raid/bear/dataset/twitter/twitter-2010.dgl"

bindata = np.fromfile(TWBINPATH,dtype=np.int32)
src = bindata[1::2]
dst = bindata[::2]
graph = dgl.graph((src,dst))
dgl.save_graphs(TWDGLPATH, [graph])
