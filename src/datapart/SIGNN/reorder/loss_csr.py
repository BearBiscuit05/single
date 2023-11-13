import torch
import dgl
import numpy as np
import time

def featSlice(FEATPATH,beginIndex,endIndex,featLen):
    blockByte = 4 # float32 4byte
    offset = (featLen * beginIndex) * blockByte
    subFeat = np.fromfile(FEATPATH, dtype=np.float32, count=(endIndex - beginIndex) * featLen, offset=offset)
    return subFeat.reshape(-1,featLen)

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

def loss_csr(FILEPATH,featLen,sliceNUM,lossNode):
    PTRPATH = FILEPATH +f"/indptr.bin"
    INDICEPATH = FILEPATH +f"/indices.bin"
    FEATPATH = FILEPATH +f"/feat.bin"
    
    raw_ptr = torch.tensor(np.fromfile(PTRPATH,dtype=np.int32))
    nodeNUM = raw_ptr.shape[0] - 1
    # ptr loss_method
    ptr_diff = torch.diff(raw_ptr) 
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
        allNode = torch.arange(nodeNUM).to(torch.int32)
        mask = torch.ones(ptr_diff.size(0), dtype=torch.bool)
        mask[lossNode.to(torch.int64)] = False
        node_save_idx = torch.masked_select(allNode, mask).to(torch.int32)
        allNode = None
        
    ptr_diff[~mask] = 0
    new_ptr = torch.cat((torch.zeros(1).to(torch.int32),torch.cumsum(ptr_diff,dim = 0).to(torch.int32)))
    id2featMap = mask.cumsum(dim=0).to(torch.int32)
    id2featMap -= 1
    id2featMap[~mask] = -1
    ptr_diff,mask = None,None

    # indice
    raw_indice = torch.tensor(np.fromfile(INDICEPATH,dtype=np.int32))
    new_indice = torch.zeros(new_ptr[-1].item()).to(torch.int32)
    raw_ptr,new_ptr = raw_ptr.cuda(),new_ptr.cuda()
    raw_indice,new_indice = raw_indice.cuda(),new_indice.cuda()
    start = time.time()
    dgl.loss_csr(raw_ptr,new_ptr,raw_indice,new_indice)
    print(f"all loss csr time {time.time() - start : .4f}")
    raw_ptr,raw_indice = None,None

    # feat
    boundList = genSliceBound(sliceNUM,new_ptr.numel())
    idsSliceList = sliceIds(node_save_idx,boundList)
    feat_cuda = torch.zeros([node_save_idx.numel(),100],dtype=torch.float32).cuda()

    start = time.time()
    offset = 0
    for sliceIndex in range(sliceNUM):
        beginIdx = boundList[sliceIndex]
        endIdx = boundList[sliceIndex+1]
        sliceFeat = torch.tensor(featSlice(FEATPATH,beginIdx,endIdx,featLen))       
        choice_ids = idsSliceList[sliceIndex] - boundList[sliceIndex]
        sliceSize = choice_ids.shape[0]
        add_feat = sliceFeat.cuda()[choice_ids.to(torch.int64)]
        feat_cuda[offset:offset+sliceSize] = add_feat
        offset += sliceSize
    print(f"all feat slice time {time.time() - start : .4f}")
    return new_ptr,raw_indice,feat_cuda,id2featMap

if __name__ == '__main__':
    FILEPATH = "/home/bear/workspace/single-gnn/data/partition/PD/part0"
    featLen = 100
    # 有损csr函数测试，传入文件路径，遵循ptr,indice,feat的修改顺序
    loss_csr(FILEPATH,featLen,4,None)