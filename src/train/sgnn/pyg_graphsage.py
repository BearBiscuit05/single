import copy
import os
import os.path as osp
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch_geometric.datasets import Reddit
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv
import sys
import time
current_folder = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_folder+"/../../"+"load")
from loader import CustomDataset

class SAGE(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int,
                 out_channels: int, num_layers: int = 2):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))
        self.num_layers = num_layers

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = x.relu_()
                x = F.dropout(x, p=0.5, training=self.training)
        return x

    def inference(self, x_all,subgraph_loader):
        pbar = tqdm(total=x_all.size(0) * self.num_layers)
        pbar.set_description('Evaluating')

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        for i in range(self.num_layers):
            xs = []
            for batch in subgraph_loader:
                x = x_all[batch.n_id].to("cuda:0")
                edge_index = batch.edge_index.to("cuda:0")
                x = self.convs[i](x, edge_index)
                x = x[:batch.batch_size]
                if i != self.num_layers - 1:
                    x = x.relu()
                xs.append(x.cpu())

                pbar.update(batch.batch_size)

            x_all = torch.cat(xs, dim=0)

        pbar.close()

        return x_all

@torch.no_grad()
def test(model,evaluator,data,subgraph_loader,split_idx):
    model.eval()

    out = model.inference(data.x,subgraph_loader)

    y_true = data.y.cpu()
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': y_true[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    val_acc = evaluator.eval({
        'y_true': y_true[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y_true[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, val_acc, test_acc

def run(rank, world_size, dataset):
    train_loader = DataLoader(dataset=dataset, batch_size=1024, collate_fn=collate_fn)#,pin_memory=True)
    torch.manual_seed(12345)
    model = SAGE(100, 256, 47).to('cuda:0')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(0, dataset.epoch):
        startTime = time.time()
        model.train()    
        for graph,feat,label,number in train_loader:        
            optimizer.zero_grad()    
            # print(graph) 
            out = model(feat.to('cuda:0'), graph)[:number]
            loss = F.cross_entropy(out, label[:number].to(torch.int64).to('cuda:0'))
            loss.backward()
            optimizer.step()
        runTime = time.time() - startTime
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Time: {runTime:.3f}s')

    root = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'dataset')
    dataset = PygNodePropPredDataset('ogbn-products', root)
    split_idx = dataset.get_idx_split()
    evaluator = Evaluator(name='ogbn-products')
    data = dataset[0].to("cuda:0", 'x', 'y')
    subgraph_loader = NeighborLoader(
        data,
        input_nodes=None,
        num_neighbors=[-1],
        batch_size=4096,
        num_workers=12,
        persistent_workers=True,
    )

    train_acc, val_acc, test_acc = test(model,evaluator,data,subgraph_loader,split_idx)
    print(f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, '
                  f'Test: {test_acc:.4f}')

def collate_fn(data):
    return data[0]

if __name__ == '__main__':
    world_size = 1
    print('Let\'s use', world_size, 'GPUs!')
    dataset = CustomDataset("./../../../config/pyg_products_graphsage.json")
    run(0, world_size, dataset)

    
