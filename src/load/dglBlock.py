import dgl
import torch
from dgl.heterograph import DGLBlock

# g = dgl.graph((torch.tensor([0, 1]), torch.tensor([1, 2])))
#gidx = dgl.heterograph_index.create_unitgraph_from_coo

def create_dgl_block(data, num_src_nodes, num_dst_nodes):
    row, col = data
    gidx = dgl.heterograph_index.create_unitgraph_from_coo(2, num_src_nodes, num_dst_nodes, row, col, 'coo')
    g = DGLBlock(gidx, (['_N'], ['_N']), ['_E'])
    return g

if __name__ == "__main__":
    row = torch.tensor([0, 1, 2])
    col = torch.tensor([1, 2, 3])

    block = create_dgl_block((row,col),3,4)
    # g = dgl.to_block(g)
    print(block)