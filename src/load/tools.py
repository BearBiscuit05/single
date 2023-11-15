import torch
import dgl
import numpy as np
import time

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

def countMemToLoss(edgeNUM,nodeNUM,featLen,ratedMem):
    int32Byte = 4
    int64Byte = 8
    float32Byte = 4
    MemNeed = (edgeNUM + nodeNUM + 1) * int32Byte + nodeNUM * featLen * float32Byte + nodeNUM * int64Byte
    MemNeed_GB = MemNeed / (1024 ** 3)
    #print(f"Estimated Memory Needed: {MemNeed_GB:.2f} GB")
    if ratedMem >= MemNeed:
        return False
    else:
        #print("need loss graph")
        return True

def loss_csr(raw_ptr,raw_indice,lossNode,saveNode):
    start = time.time()
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
        # mask = torch.ones(nodeNUM, dtype=torch.bool).cuda()
        # mask[lossNode.to(torch.int64)] = False
        node_save_idx = saveNode
    
    #ptr_diff[lossNode.to(torch.int64)] = 0
    condition = ptr_diff > 100
    ptr_diff[condition] = (ptr_diff[condition] * 0.7).to(torch.int32) 
    
    new_ptr = torch.cat((torch.zeros(1).to(torch.int32).cuda(),torch.cumsum(ptr_diff,dim = 0).to(torch.int32))).cuda()
    #id2featMap = mask.cumsum(dim=0).to(torch.int32).cuda()
    # id2featMap -= 1
    # id2featMap[lossNode.to(torch.int64)] = -1
    id2featMap = 0
    ptr_diff,mask = None,None

    
    # indice
    new_indice = raw_indice.clone()[:new_ptr[-1].item()]
    dgl.loss_csr(raw_ptr,new_ptr,raw_indice,new_indice)
    print("raw_indice len :",len(raw_indice))
    print("new_indice len :",len(new_indice))
    raw_ptr,raw_indice = None,None
    return new_ptr,new_indice,id2featMap

def loss_feat(raw_feat, sliceNUM, id2featMap, featLen):
    # from cup to gpu with loss
    node_save_idx = torch.nonzero(id2featMap.cpu() >= 0).reshape(-1).to(torch.int32)
    boundList = genSliceBound(sliceNUM, node_save_idx.numel())
    idsSliceList = sliceIds(node_save_idx, boundList)
    
    loss_feat = torch.zeros([node_save_idx.numel(), featLen], dtype=torch.float32).cuda()
    # start = time.time()
    offset = 0
    for sliceIndex in range(sliceNUM):
        beginIdx = boundList[sliceIndex]
        endIdx = boundList[sliceIndex + 1]
        sliceFeat = torch.tensor(featSlice(raw_feat, beginIdx, endIdx, featLen))
        
        choice_ids = idsSliceList[sliceIndex] - boundList[sliceIndex]
        
        sliceSize = choice_ids.shape[0]
        add_feat = sliceFeat.cuda()[choice_ids.to(torch.int64)]
        loss_feat[offset:offset + sliceSize] = add_feat
        offset += sliceSize
    
    # print(f"Total feat slice time {time.time() - start : .4f}")
    # print(f"Final loss_feat shape: {loss_feat.shape}")
    return loss_feat
