import torch
import dgl
import numpy as np
import time
from memory_profiler import profile
import gc

def featSlice(raw_feat,beginIndex,endIndex,featLen):
    blockByte = 4 # float32 4byte
    offset = (featLen * beginIndex) * blockByte
    #subFeat = np.fromfile(FEATPATH, dtype=np.float32, count=(endIndex - beginIndex) * featLen, offset=offset).reshape(-1,featLen)
    subFeat = raw_feat[beginIndex:endIndex]
    return subFeat

def sliceIds(Ids,sliceTable):
    beginIndex = 0
    ans = []
    for tar in sliceTable[1:]:
        position = torch.searchsorted(Ids, tar)
        slice = Ids[beginIndex:position]
        ans.append(slice)
        beginIndex = position
    return ans

def genSliceBound(sliceNUM,nodeNUM):
    slice = nodeNUM // sliceNUM + 1
    boundList = [0]
    start = slice
    for i in range(sliceNUM):
        boundList.append(start)
        start += slice
    boundList[-1] = nodeNUM
    return boundList

def countMemToLoss(edgeNUM,nodeNUM,featLen,ratedMem,printInfo=False):
    int32Byte, int64Byte, float32Byte = 4, 8, 4
    MemNeed_B = (edgeNUM + nodeNUM + 1) * int32Byte + nodeNUM * featLen * float32Byte + nodeNUM * int64Byte
    MemNeed_KB = MemNeed_B / (1024 ** 1)
    MemNeed_MB = MemNeed_B / (1024 ** 2)
    MemNeed_GB = MemNeed_B / (1024 ** 3)
    if printInfo:
        print(f"Basic Parameters:")
        print(f"Number of Edges: {edgeNUM},Number of Nodes: {nodeNUM},Feature Length: {featLen}")
        print(f"Memory Needed: {MemNeed_GB:.2f} GB/{MemNeed_MB:.2f} MB/{MemNeed_KB:.2f} KB/{MemNeed_B:.2f} B")
    if ratedMem >= MemNeed_B:
        return False
    else:
        return True

#@profile
def loss_csr(raw_ptr,raw_indice,lossNode,saveNode):
    nodeNUM = raw_ptr.shape[0] - 1
    ptr_diff = torch.diff(raw_ptr).cuda()
    
    if lossNode == None:
        num_elements = int(raw_ptr.shape[0]*0.9)
        min_value = 0
        max_value = nodeNUM
        node_save_idx = torch.randint(min_value, max_value - 1, (num_elements,), dtype=torch.int32)
        
        node_save_idx = torch.unique(node_save_idx)
        node_save_idx,_ = torch.sort(node_save_idx)  
        mask = torch.zeros(ptr_diff.size(0), dtype=torch.bool)
        mask[node_save_idx.to(torch.int64)] = True
    else:
        mask = torch.ones(nodeNUM, dtype=torch.bool).cuda()
        mask[lossNode.to(torch.int64)] = False

    ptr_diff[lossNode.to(torch.int64)] = 0
    # condition = ptr_diff > 100
    # ptr_diff[condition] = (ptr_diff[condition] * 0.7).to(torch.int32) 
    
    new_ptr = torch.cat((torch.zeros(1).to(torch.int32).cuda(),torch.cumsum(ptr_diff,dim = 0).to(torch.int32))).cuda()
    id2featMap = mask.cumsum(dim=0).to(torch.int32).cuda()
    id2featMap -= 1
    id2featMap[lossNode.to(torch.int64)] = -1
    ptr_diff,mask = None,None

    new_indice = raw_indice.clone()[:new_ptr[-1].item()]
    dgl.loss_csr(raw_ptr,new_ptr,raw_indice,new_indice)
    raw_ptr,raw_indice = None,None
    emptyCache()
    return new_ptr,new_indice,id2featMap


def streamLossGraph(raw_ptr,raw_indice,lossNode,sliceNUM=1,randomLoss=0.5,degreeCut=None,CutRatio=0.5):
    # raw_ptr,new_ptr始终位于GPU中，indice同样位于GPU中，raw流式传入
    raw_ptr = raw_ptr.cuda()
    raw_indice = raw_indice.cpu()
    nodeNUM = raw_ptr.shape[0] - 1
    ptr_diff = torch.diff(raw_ptr)  # 0.01s

    # 裁剪点 0.2s
    length = lossNode.size(0)
    selected_indices = torch.randperm(length)[:int(length * randomLoss)]
    lossNode = lossNode[selected_indices]
    mask = torch.ones(nodeNUM, dtype=torch.bool).cuda()
    mask[lossNode.to(torch.int64)] = False
    ptr_diff[lossNode.to(torch.int64)] = 0

    print(f"loss... 节点数{nodeNUM},裁剪数{lossNode.shape[0]}")

    # 裁剪边
    if degreeCut != None:
        condition = ptr_diff >= degreeCut
        ptr_diff[condition] = (ptr_diff[condition] * CutRatio).to(torch.int32) 

    # allTime = time.time()
    new_ptr = torch.cat((torch.zeros(1).to(torch.int32).to(ptr_diff.device),torch.cumsum(ptr_diff,dim = 0).to(torch.int32)))
    id2featMap = mask.cumsum(dim=0).to(torch.int32)
    id2featMap -= 1
    id2featMap[lossNode.to(torch.int64)] = -1
    # print("id2featMap shape:",id2featMap.shape)
    # print("mask shape:",mask.shape)
    # print("loss_csr max map idx",torch.max(id2featMap))
    # exit(-1)
    
    ptr_diff,mask = None,None
    # print(f"ptr_diff using time :{time.time()-allTime:.3f}s")
    # indice

    blockSize = (nodeNUM - 1) // sliceNUM + 1
    bound = []
    lastIdx = 0
    for i in range(sliceNUM):
        nextSlice = min((i+1)*blockSize,nodeNUM)
        bound.append([lastIdx,nextSlice])
        lastIdx = nextSlice

    new_indice = torch.zeros(new_ptr[-1].item()-1,dtype=torch.int32,device="cuda:0")
    # allTime = time.time()
    for left,right in bound:
        raw_off = raw_ptr[left:right+1]-raw_ptr[left].item()
        new_off = new_ptr[left:right+1]-new_ptr[left].item()
        rawIndiceOff = raw_indice[raw_ptr[left].item():raw_ptr[right].item()].cuda()
        newIndiceOff = new_indice[new_ptr[left].item():new_ptr[right].item()]
        dgl.loss_csr(raw_off,new_off,rawIndiceOff,newIndiceOff)
    # print(f"loss_csr func using time :{time.time()-allTime:.3f}s")
    raw_ptr,raw_indice = None,None
    return new_ptr,new_indice,id2featMap

#@profile
def loss_feat(loss_feat,raw_feat, sliceNUM, id2featMap, featLen,device):
    # from cup to gpu with loss
    node_save_idx = torch.nonzero(id2featMap.cpu() >= 0).reshape(-1).to(torch.int32)
    boundList = genSliceBound(sliceNUM, node_save_idx.numel())
    idsSliceList = sliceIds(node_save_idx, boundList)
    #print(f"Preprocess time: {time.time() - start_preprocess : .4f} seconds")
    
    offset = 0
    for sliceIndex in range(sliceNUM):
        beginIdx = boundList[sliceIndex]
        endIdx = boundList[sliceIndex + 1]
        
        #start_slice = time.time()      
        sliceFeat = torch.as_tensor(featSlice(raw_feat, beginIdx, endIdx, featLen))
        choice_ids = idsSliceList[sliceIndex] - boundList[sliceIndex]
        sliceSize = choice_ids.shape[0]
        loss_feat[offset:offset + sliceSize] = sliceFeat.to(device)[choice_ids.to(torch.int64)]
        offset += sliceSize
        #print(f"Slice {sliceIndex + 1} time: {time.time() - start_slice : .4f} seconds")
    
    #print(f"Total feat slice time: {time.time() - start_loss_feat : .4f} seconds")
    #print('-'*20)
    # return loss_feat

def featAdd(addIdx, addfeat, memfeat, cudafeat):
    #转换addFeat，但是需要注意mem和cuda两个feat向量的分开转换
    #总体可以视为feat[addIdx] = addfeat
    #分成memfeat[addIdx_mem] = addfeat_mem; cudafeat[addIdx_cuda] = addfeat_cuda
    #此处addIdx已经通过map转为实际索引位置
    start = time.time()
    addIdx = addIdx.cuda()

    addIdx_mem_indice = torch.nonzero(addIdx < 0).reshape(-1)
    addIdx_cuda_indice = torch.nonzero(addIdx > 0).reshape(-1)

    addIdx_mem = addIdx[addIdx_mem_indice] * (-1)
    addIdx_cuda = addIdx[addIdx_cuda_indice]
    
    #这里有点慢
    memfeat[addIdx_mem] = addfeat[addIdx_mem_indice]
    cudafeat[addIdx_cuda] = addfeat[addIdx_cuda_indice].cuda()

    print(f"featAdd {time.time() - start:.4f}s")

#test:
# addIdx = torch.tensor([-1,-3,2,4,1,3,-2])
# memfeat = torch.tensor([-1,5,6,7,8,9,10])
# cudafeat = torch.tensor([-1,11,12,13,14,15,16]).cuda()
# addfeat = torch.tensor([20,21,22,23,24,25,26])
# featAdd(addIdx, addfeat, memfeat,cudafeat)

def init_cac(lossMap, feat, memfeat, cudafeat, map):
    #将feat初始化迁移到memfeat和cudafeat
    mask = torch.ones(lossMap.shape[0], dtype = torch.bool, device='cuda')
    mask[lossMap == -1] = False

    cutfeat = feat[~mask]
    memfeat[1 : cutfeat.shape[0] + 1] = cutfeat
    map[torch.nonzero(~mask).reshape(-1)] = (-1) * torch.arange(1, cutfeat.shape[0] + 1, device = 'cuda', dtype=torch.int64)
    cutfeat = None

    savefeat = feat[mask]
    cudafeat[1 :savefeat.shape[0] + 1] = savefeat
    map[torch.nonzero(mask).reshape(-1)] = torch.arange(1, savefeat.shape[0] + 1, device = 'cuda', dtype=torch.int64)
    savefeat = None
    
#cuda and cpu 转换
def loss_feat_cac(lossMap, memfeat, cudafeat, map):
    #调用该函数需要保证
    #1.memfeat和cudafeat中含有所有当前子图的feat(可以有冗余feat但是必需全有)
    #2.lossNode存的是节点索引，map作用是 节点索引 -> memfeat/cudafeat索引
    #作用: 将所有当前图节点中，未被loss的节点的feat全部上cudafeat，并且更新map
    #即：交换cudafeat中的非saveNode节点 和 memfeat中的saveNode节点 的位置并且维护map
    #需要保证：cudafeat中非saveNode节点数 > memfeat中saveNode节点数
    emptyCache()
    start = time.time()
    saveNode = torch.nonzero(lossMap != -1).reshape(-1)

    #获取mem中的saveNode
    saveIdxMap_mem = saveNode[torch.nonzero(map[saveNode] < 0).reshape(-1)]
    saveIdx_mem = map[saveIdxMap_mem] * (-1)

    #获取cuda中的非save的索引
    mask = torch.ones(map.shape[0], dtype=torch.bool, device='cuda')
    mask[saveNode] = False
    mask[torch.nonzero(map < 0).reshape(-1)] = False
    nsaveMap_cuda = (torch.nonzero(mask).reshape(-1))[:saveIdx_mem.shape[0]]
    nsaveIdx_cuda = map[nsaveMap_cuda]

    #将nsave_cuda[:len(save_mem)]和save_mem交换并维护map
    cuda_tmp = cudafeat[nsaveIdx_cuda]
    cudafeat[nsaveIdx_cuda] = memfeat[saveIdx_mem].cuda()
    memfeat[saveIdx_mem] = cuda_tmp.cpu()

    #维护map
    map_cuda_tmp = map[nsaveMap_cuda]
    map[nsaveMap_cuda] = map[saveIdxMap_mem]
    map[saveIdxMap_mem] = map_cuda_tmp
    print(f"loss_feat_cac {time.time() - start:.4f}s")

#test:
# map = torch.tensor([-1,-2,1,2,3,-3,-4,-5,4,5,6])
# lossMap = torch.tensor([1,1,1,1,-1,-1,1,1])
# memfeat = torch.tensor([-1,11,12,13,14,15])
# cudafeat = torch.tensor([-1,21,22,23,24,25,26,27]).cuda()
# loss_feat_cac(lossMap,memfeat,cudafeat,map)


def emptyCache():
    torch.cuda.empty_cache()
    gc.collect()


def print_gpu_memory(index):
    # 打印显存信息
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(index)
        memory_allocated = torch.cuda.memory_allocated(index) / 1024 ** 3  # 转换为GB
        memory_cached = torch.cuda.memory_cached(index) / 1024 ** 3  # 转换为GB
        print(f"GPU {index}: {gpu}")
        print(f"  Allocated Memory: {memory_allocated:.2f} GB")
        print(f"  Cached Memory: {memory_cached:.2f} GB")
    else:
        print("No GPU available.")
    

def streamAssign(rawValues,raplaceIdx,replaceValue,sliceNUM=4):
    # 流式传递显存
    # rawValues     位于cuda里面
    # replaceValue  位于内存中
    idxBatches = torch.chunk(raplaceIdx, sliceNUM, dim=0)
    valueBatches = torch.chunk(replaceValue, sliceNUM, dim=0)
    batch = [idxBatches, valueBatches]
    for idx,value in zip(*batch):
        idx = idx.cuda().to(torch.int64)
        value = value.cuda()
        rawValues[idx] = value