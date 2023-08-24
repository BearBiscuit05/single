import torch
import dgl
src_nodes = [0, 2, 3, 4, 0, 6, 7, 9, 0, 1, 2, 3, 5, 6, 7, 8, 9, 0, 4, 6, 8]
dst_nodes = [1, 3, 4, 0, 5, 7, 8, 0, 6, 2, 4, 8, 6, 7, 1, 3, 9, 5, 5, 7, 9]

src_tensor = torch.tensor(src_nodes)
dst_tensor = torch.tensor(dst_nodes)

g = dgl.graph((torch.cat([src_tensor, dst_tensor]), torch.cat([dst_tensor, src_tensor])))
print(g.edges())
E = len(src_tensor)
reverse_eids = torch.cat([torch.arange(E, 2 * E), torch.arange(0, E)])

neg_sampler = dgl.dataloading.negative_sampler.Uniform(1)

train_eid = torch.Tensor([0,2]).to(torch.int64)
sampler = dgl.dataloading.as_edge_prediction_sampler(
        dgl.dataloading.NeighborSampler([3]),
        exclude='reverse_id', reverse_eids=reverse_eids,
        negative_sampler=neg_sampler)
dataloader = dgl.dataloading.DataLoader(
        g, train_eid, sampler,
        batch_size=2, shuffle=True, drop_last=False, num_workers=1)
        
for input_nodes, pos_pair_graph, neg_pair_graph, blocks in dataloader:
    print("input_nodes:",input_nodes)
    print("pair_graph:",pos_pair_graph.edges())
    print("pair_graph edge::",pos_pair_graph.ndata[dgl.NID])
    print("neg_pair_graph:",neg_pair_graph.edges())
    print("neg_pair_graph edge:",neg_pair_graph.ndata[dgl.NID])
    print("blocks:",blocks)
    print("blocks edges:",blocks[0].edges())
    print("blocks edges:",blocks[0].edata[dgl.EID])


