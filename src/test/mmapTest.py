import numpy as np
import torch
import mmap
import time
# def gen_format_file(rank,Wsize,dataPath,datasetName,savePath):
#     graph_dir = dataPath
#     part_config = graph_dir + "/"+datasetName +'.json'
#     print('loading partitions')
#     subg, node_feat, _, gpb, _, node_type, _ = dgl.distributed.load_partition(part_config, rank)
#     node_type = node_type[0]
#     featInfo = node_feat[node_type + '/features']
#     np_featInfo = featInfo.detach().numpy()
#     return featInfo[:10]

def mmap_read(head,featLen,trainIDs):
    float_size = np.dtype(np.float32).itemsize
    start = time.time()
    feats = torch.zeros((len(trainIDs), featLen), dtype=torch.float32)
    for index, nodeID in enumerate(trainIDs):
        feat = torch.frombuffer(head, dtype=torch.float32, offset=nodeID*featLen*float_size, count=featLen)
        feats[index] = feat
    print("mmap run time :{}s".format(time.time() - start))
    return feats

def mem_read(filePath,trainIDs):
    arr = np.fromfile(filePath+"/feat.bin", dtype=np.float32)
    #arr = torch.from_file(filePath+"/feat.bin",dtype = torch.float32).view((-1,100))
    arr = torch.from_numpy(arr).reshape(-1,100)
    print(arr)
    # feats = arr.reshape(-1,100)
    # feats = torch.tensor(feats)
    start = time.time()
    #ans = torch.zeros((len(trainIDs), 100), dtype=torch.float32)
    ans = arr[trainIDs]
    # for index, nodeID in enumerate(trainIDs):
    #     ans[index] = feats[nodeID]
    print("mem run time :{}s".format(time.time() - start))

if __name__ == '__main__':
    trainIDs = np.fromfile("./TestData/ids.bin", dtype=np.int64)
    trainIDs = trainIDs.astype(np.int32)
    for index,ids in enumerate(trainIDs):
        if trainIDs[index] > 602200:
            trainIDs[index] = 602200 - index
    filePath = "../../data/products/part0"
    file = open(filePath+"/feat.bin", "r+b")
    head = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_DEFAULT)
    feats = torch.frombuffer(head, dtype=torch.float32).reshape(-1,100)
    start = time.time()
    ans = feats[trainIDs]
    #print(ans)
    print("mmap run time :{}s".format(time.time() - start))
    # t2 = mmap_read(head,100,trainIDs)
    mem_read(filePath,trainIDs)
    