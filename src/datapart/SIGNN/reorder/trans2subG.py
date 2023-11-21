import numpy as np
import dgl
import torch
import os
import copy
import gc
import time
from scipy.sparse import csr_matrix,coo_matrix
import json
from tools import *
from memory_profiler import profile
import sys

# =============== 1.partition

def acc_ana(tensor):
    num_ones = torch.sum(tensor == 1).item()  
    total_elements = tensor.numel()  
    percentage_ones = (num_ones / total_elements) * 100 
    print(f"only use by one train node : {percentage_ones:.2f}%")
    num_greater_than_1 = torch.sum(tensor > 1).item() 
    percentage_greater_than_1 = (num_greater_than_1 / total_elements) * 100
    print(f"use by multi train nodes : {percentage_greater_than_1:.2f}%")
    # edgeNUM = edgeTable.cpu().sum() - edgeNUM
    # print(f"edge add to subG : {edgeNUM} , {edgeNUM * 1.0 / allEdgeNUM * 100 :.2f}% of total edges")
    # print(f"after {index} BFS has {torch.nonzero(nodeTable).size(0)} nodes, "
    # f"{torch.nonzero(nodeTable).size(0) * 1.0 / maxID * 100 :.2f}% of total nodes")

RUNTIME = 0
SAVETIME = 0
MERGETIME = 0
MAXEDGE = 100000000

## bfs 遍历获取基础子图
def analysisG(graph,maxID,trainId=None,savePath=None):
    global RUNTIME
    global SAVETIME
    print(f"analysisG train len:{trainId.shape}")
    dst = torch.as_tensor(graph[::2])
    src = torch.as_tensor(graph[1::2])
    if trainId == None:
        trainId = torch.arange(int(maxID*0.01),dtype=torch.int64)
    nodeTable = torch.zeros(maxID,dtype=torch.int32)
    nodeTable[trainId] = 1

    batch_size = len(src) // MAXEDGE + 1
    src_batches = torch.chunk(src, batch_size, dim=0)
    dst_batches = torch.chunk(dst, batch_size, dim=0)
    batch = [src_batches, dst_batches]

    repeats = 3
    start = time.time()
    edgeTable = torch.zeros_like(src,dtype=torch.int32)
    tmpEtable = torch.zeros_like(dst_batches[0],dtype=torch.int32).cuda()
    for index in range(1,repeats+1):
        acc_table = torch.zeros_like(nodeTable,dtype=torch.int32)
        offset = 0
        for src_batch,dst_batch in zip(*batch):
            tmp_nodeTable = copy.deepcopy(nodeTable)
            tmp_nodeTable = tmp_nodeTable.cuda()
            src_batch = src_batch.cuda()
            dst_batch = dst_batch.cuda()
            tmpEtable.fill_(0)
            dgl.fastFindNeigEdge(tmp_nodeTable,tmpEtable,src_batch, dst_batch)
            edgeTable[offset:offset+len(src_batch)] = tmpEtable.cpu()[:len(src_batch)]
            offset += len(src_batch)
            tmp_nodeTable = tmp_nodeTable.cpu()
            acc_table = acc_table | tmp_nodeTable
        nodeTable = acc_table
    tmpEtable = None
    graph = graph.reshape(-1,2)
    nodeSet =  torch.nonzero(nodeTable).reshape(-1).to(torch.int32)
    edgeTable = torch.nonzero(edgeTable).reshape(-1).to(torch.int32)
    selfLoop = np.repeat(nodeSet.to(torch.int32), 2)
    subGEdge = graph[edgeTable]
    RUNTIME += time.time()-start

    saveTime = time.time()
    checkFilePath(savePath)
    DataPath = savePath + f"/raw_G.bin"
    TrainPath = savePath + f"/raw_trainIds.bin"
    NodePath = savePath + f"/raw_nodes.bin"
    saveBin(nodeSet,NodePath)
    saveBin(selfLoop,DataPath)
    saveBin(subGEdge,DataPath,addSave=True)
    saveBin(trainId,TrainPath)
    SAVETIME += time.time()-saveTime

#@profile
def PRgenG(RAWPATH,nodeNUM,partNUM,savePath=None):
    GRAPHPATH = RAWPATH + "/graph.bin"
    TRAINPATH = RAWPATH + "/trainIds.bin"

    for i in range(partNUM):
        PATH = savePath + f"/part{i}" 
        checkFilePath(PATH)
    
    graph = torch.from_numpy(np.fromfile(GRAPHPATH,dtype=np.int32))
    src,dst = graph[::2],graph[1::2]
    trainIds = torch.from_numpy(np.fromfile(TRAINPATH,dtype=np.int64))
    edgeTable = torch.zeros_like(src).to(torch.int32)
    template_array = torch.zeros(nodeNUM,dtype=torch.int32)

    batch_size = len(src) // MAXEDGE + 1
    src_batches = torch.chunk(src, batch_size, dim=0)
    dst_batches = torch.chunk(dst, batch_size, dim=0)
    batch = [src_batches, dst_batches]

    inNodeTable, outNodeTable = template_array.clone().cuda(), template_array.clone().cuda()
    for src_batch,dst_batch in zip(*batch):
        src_batch = src_batch.cuda()
        dst_batch = dst_batch.cuda()
        inNodeTable,outNodeTable = dgl.sumDegree(inNodeTable,outNodeTable,src_batch,dst_batch)
    outNodeTable = outNodeTable.cpu() # innodeTable still in GPU for next use

    nodeValue = template_array.clone()
    nodeInfo = template_array.clone()
    nodeValue[trainIds] = 10000

    # random method
    shuffled_indices = torch.randperm(trainIds.size(0))
    trainIds = trainIds[shuffled_indices]
    trainBatch = torch.chunk(trainIds, partNUM, dim=0)
    for index,ids in enumerate(trainBatch):
        info = 1 << index
        nodeInfo[ids] = info
        # 存储训练集
        PATH = savePath + f"/part{index}" 
        TrainPath = PATH + f"/raw_trainIds.bin"
        saveBin(ids,TrainPath)
    # ====

    tmp_etable = torch.zeros_like(dst_batches[0],dtype=torch.int32).cuda()
    for _ in range(3):
        offset = 0
        acc_nodeValue = torch.zeros_like(nodeValue,dtype=torch.int32)
        acc_nodeInfo = torch.zeros_like(nodeInfo,dtype=torch.int32)
        for src_batch,dst_batch in zip(*batch):  
            batchLen = len(src_batch)
            tmp_nodeValue = nodeValue.clone().cuda()
            tmp_nodeInfo = nodeInfo.clone().cuda() 
            src_batch = src_batch.cuda()
            dst_batch = dst_batch.cuda()  
            tmp_etable.fill_(0)
            dgl.per_pagerank(dst_batch,src_batch,tmp_etable,inNodeTable,tmp_nodeValue,tmp_nodeInfo)
            edgeTable[offset:offset+batchLen] = tmp_etable[:batchLen].cpu()
            tmp_nodeValue = tmp_nodeValue.cpu()
            tmp_nodeInfo = tmp_nodeInfo.cpu()
            acc_nodeValue += tmp_nodeValue - nodeValue
            acc_nodeInfo = acc_nodeInfo | tmp_nodeInfo     
            offset += len(src_batch)
        nodeValue = nodeValue + acc_nodeValue
        nodeInfo = acc_nodeInfo
        tmp_nodeValue=None
        tmp_nodeInfo=None
    inNodeTable = None

    torch.cuda.empty_cache()
    gc.collect()
    for bit_position in range(partNUM):
        nodeIndex = (nodeInfo & (1 << bit_position)) != 0
        edgeIndex = (edgeTable & (1 << bit_position)) != 0
        nid = torch.nonzero(nodeIndex).reshape(-1).to(torch.int32)
        PATH = savePath + f"/part{bit_position}" 
        DataPath = PATH + f"/raw_G.bin"
        NodePath = PATH + f"/raw_nodes.bin"
        PRvaluePath = PATH + f"/sortIds.bin"
        selfLoop = np.repeat(trainBatch[bit_position].to(torch.int32), 2)
        saveBin(nid,NodePath)
        saveBin(selfLoop,DataPath)
        graph = graph.reshape(-1,2)
        sliceNUM = 10
        offsetSize = len(edgeIndex) // sliceNUM + 1
        offset = 0
        for i in range(sliceNUM):
            sliceLen = min((i+1)*offsetSize,len(edgeIndex))
            g_gpu = graph[offset:offset + sliceLen]
            idx_gpu = edgeIndex[offset:offset + sliceLen]
            subEdge = g_gpu.cuda()[idx_gpu.cuda()].cpu()
            saveBin(subEdge,DataPath,addSave=True)
            offset += sliceLen
        start = time.time()
        partValue = nodeValue[nodeIndex]  
        _ , sort_indice = torch.sort(partValue,dim=0,descending=True)
        sort_nodeid = nid[sort_indice]
        saveBin(sort_nodeid,PRvaluePath)

# =============== 2.graphToSub    
def nodeShuffle(raw_node,raw_graph):
    torch.cuda.empty_cache()
    gc.collect()
    srcs = raw_graph[1::2]
    dsts = raw_graph[::2]
    raw_node = convert_to_tensor(raw_node, dtype=torch.int32).cuda()
    srcs_tensor = convert_to_tensor(srcs, dtype=torch.int32)
    dsts_tensor = convert_to_tensor(dsts, dtype=torch.int32)
    
    uni = torch.ones(len(raw_node)).to(torch.int32).cuda()
    batch_size = len(srcs) // MAXEDGE + 1
    src_batches = list(torch.chunk(srcs_tensor, batch_size, dim=0))
    dst_batches = list(torch.chunk(dsts_tensor, batch_size, dim=0))

    batch = [src_batches, dst_batches]
    for index,(src_batch,dst_batch) in enumerate(zip(*batch)):
        src_batch = src_batch.cuda()
        dst_batch = dst_batch.cuda()
        srcShuffled,dstShuffled,uni = dgl.mapByNodeSet(raw_node,uni,src_batch,dst_batch)
        srcShuffled = srcShuffled.cpu()
        dstShuffled = dstShuffled.cpu()   
        src_batches[index] = srcShuffled
        dst_batches[index] = dstShuffled 
    srcs_tensor = torch.cat(src_batches)
    dsts_tensor = torch.cat(dst_batches)
    uni = uni.cpu()
    return srcs_tensor,dsts_tensor,uni

def trainIdxSubG(subGNode,trainSet):
    trainSet = torch.as_tensor(trainSet).to(torch.int32)
    Lid = torch.zeros_like(trainSet).to(torch.int32).cuda()
    dgl.mapLocalId(subGNode.cuda(),trainSet.cuda(),Lid)
    Lid = Lid.cpu().to(torch.int64)
    return Lid

def coo2csr(srcs,dsts):
    g = dgl.graph((dsts,srcs)).formats('csr')
    indptr, indices, _ = g.adj_sparse(fmt='csr')
    return indptr,indices

def rawData2GNNData(RAWDATAPATH,partitionNUM,LABELPATH):
    labels = np.fromfile(LABELPATH,dtype=np.int64)
    for i in range(partitionNUM):
        startTime = time.time()
        PATH = RAWDATAPATH + f"/part{i}" 
        rawDataPath = PATH + f"/raw_G.bin"
        rawTrainPath = PATH + f"/raw_trainIds.bin"
        rawNodePath = PATH + f"/raw_nodes.bin"
        PRvaluePath = PATH + f"/sortIds.bin"
        SubTrainIdPath = PATH + "/trainIds.bin"
        SubIndptrPath = PATH + "/indptr.bin"
        SubIndicesPath = PATH + "/indices.bin"
        SubLabelPath = PATH + "/labels.bin"
        checkFilePath(PATH)
        data = np.fromfile(rawDataPath,dtype=np.int32)
        node = np.fromfile(rawNodePath,dtype=np.int32)
        trainidx = np.fromfile(rawTrainPath,dtype=np.int64)  
        remappedSrc,remappedDst,uniNode = nodeShuffle(node,data)
        subLabel = labels[uniNode.to(torch.int64)]
        indptr, indices = cooTocsr(remappedDst,remappedSrc)
        trainidx = trainIdxSubG(uniNode,trainidx)
        saveBin(subLabel,SubLabelPath)
        saveBin(trainidx,SubTrainIdPath)
        saveBin(indptr,SubIndptrPath)
        saveBin(indices,SubIndicesPath)
        pridx = torch.tensor(np.fromfile(PRvaluePath,dtype=np.int32)).cuda()
        emp = pridx.clone()
        node = torch.as_tensor(node).cuda()
        uni = torch.ones(len(node)).to(torch.int32).cuda()
        remappedSrc,remappedDst,uni = dgl.mapByNodeSet(node,uni,pridx,emp)
        remappedSrc = remappedSrc.cpu()
        saveBin(remappedSrc,PRvaluePath)
        print(f"map data time : {time.time()-startTime:.4f}s")
        startTime = time.time()

# =============== 3.featTrans
def featSlice(FEATPATH,beginIndex,endIndex,featLen):
    blockByte = 4 # float32 4byte
    offset = (featLen * beginIndex) * blockByte
    subFeat = torch.as_tensor(np.fromfile(FEATPATH, dtype=np.float32, count=(endIndex - beginIndex) * featLen, offset=offset))
    return subFeat.reshape(-1,featLen)

def sliceIds(Ids,sliceTable):
    beginIndex = 0
    ans = []
    for tar in sliceTable[1:]:
        print(Ids)
        position = torch.searchsorted(Ids, tar)
        slice = Ids[beginIndex:position]
        ans.append(slice)
        beginIndex = position
    return ans

def genSubGFeat(SAVEPATH,FEATPATH,partNUM,nodeNUM,sliceNUM,featLen):
    # 获得切片
    torch.cuda.empty_cache()
    gc.collect()
    slice = nodeNUM // sliceNUM + 1
    boundList = [0]
    start = slice
    for i in range(sliceNUM):
        boundList.append(start)
        start += slice
    boundList[-1] = nodeNUM

    idsSliceList = [[] for i in range(partNUM)]
    for i in range(partNUM):
        file = SAVEPATH + f"/part{i}/raw_nodes.bin"
        ids = torch.as_tensor(np.fromfile(file,dtype=np.int32))
        idsSliceList[i] = sliceIds(ids,boundList)
    
    for sliceIndex in range(sliceNUM):
        beginIdx = boundList[sliceIndex]
        endIdx = boundList[sliceIndex+1]
        sliceFeat = featSlice(FEATPATH,beginIdx,endIdx,featLen).cuda()
        for index in range(partNUM):
            fileName = SAVEPATH + f"/part{index}/feat.bin"
            SubIdsList = idsSliceList[index][sliceIndex]
            t_SubIdsList = SubIdsList - beginIdx
            subFeat = sliceFeat[t_SubIdsList.to(torch.int64).cuda()]
            subFeat = subFeat.cpu()
            saveBin(subFeat,fileName,addSave=sliceIndex)

def genAddFeat(beginId,addIdx,SAVEPATH,FEATPATH,partNUM,nodeNUM,sliceNUM,featLen):
    torch.cuda.empty_cache()
    gc.collect()
    slice = nodeNUM // sliceNUM + 1
    boundList = [0]
    start = slice
    for i in range(sliceNUM):
        boundList.append(start)
        start += slice
    boundList[-1] = nodeNUM

    file = SAVEPATH + f"/part{beginId}/raw_nodes.bin"
    ids = torch.as_tensor(np.fromfile(file,dtype=np.int32))
    addIdx.append(ids)
    print(addIdx)
    exit
    for i in range(partNUM+1):
        addIdx[i] = sliceIds(addIdx[i],boundList)

    for sliceIndex in range(sliceNUM):
        beginIdx = boundList[sliceIndex]
        endIdx = boundList[sliceIndex+1]
        sliceFeat = featSlice(FEATPATH,beginIdx,endIdx,featLen).cuda()
        for index in range(partNUM + 1):
            if index == partNUM:
                fileName = SAVEPATH + f"/part{beginId}/test_feat.bin"
            else:
                fileName = SAVEPATH + f"/part{index}/test_addfeat.bin"
            SubIdsList = addIdx[index][sliceIndex]
            t_SubIdsList = SubIdsList - beginIdx
            subFeat = sliceFeat[t_SubIdsList.to(torch.int64).cuda()]
            subFeat = subFeat.cpu()
            saveBin(subFeat,fileName,addSave=sliceIndex)


# =============== 4.addFeat
cur ,res = [] ,[]
cur_sum, res_sum = 0, -1

def dfs(part_num,diffMatrix):
    global cur,res,cur_sum,res_sum
    if (len(cur) == part_num):
        if res_sum == -1:
            res = cur[:]
            res_sum = cur_sum
        elif cur_sum < res_sum:
            res_sum = cur_sum
            res = cur[:]
            print(res,res_sum)
        return
    for i in range(0, part_num):
        if (i in cur or (res_sum != -1 and len(cur) > 0 and cur_sum + diffMatrix[cur[-1]][i] > res_sum)):
            continue
        if len(cur) != 0:
            cur_sum += diffMatrix[cur[-1]][i] 
        cur.append(i)
        dfs(part_num,diffMatrix)
        cur = cur[:-1]
        if len(cur) != 0:
            cur_sum -= diffMatrix[cur[-1]][i]

def cal_min_path(diffMatrix, nodesList, part_num, base_path):
    base_path += '/part'
    start = time.time()
    maxNodeNum = 0
    for i in range(part_num):
        path = base_path + str(i) + '/raw_nodes.bin'
        nodes = torch.as_tensor(np.fromfile(path, dtype=np.int32)).cuda()
        maxNodeNum = max(maxNodeNum, nodes.shape[0])
        nodesList.append(nodes)
    
    res1 = torch.zeros(maxNodeNum, dtype=torch.int32).cuda()
    res2 = torch.zeros(maxNodeNum, dtype=torch.int32).cuda()
    print(f"加载所有节点{time.time() - start:.4f}s")
    for i in range(part_num):
        for j in range(i + 1,part_num):
            node1 = nodesList[i]
            node2 = nodesList[j]
            res1.fill_(0)
            res2.fill_(0)
            dgl.findSameNode(node1, node2, res1, res2)
            sameNum = torch.sum(res1).item()
            diffMatrix[i][j] = node2.shape[0] - sameNum
            diffMatrix[j][i] = node1.shape[0] - sameNum

            print("part{} shape:{},part{} shape:{}, 相同的节点数:{}".format(i,node1.shape,j,node2.shape,sameNum))

    start = time.time()
    dfs(part_num ,diffMatrix)
    print("dfs 用时{}".format(time.time() - start))
    return maxNodeNum, res

def genFeatIdx(part_num, base_path, nodeList, part_seq, featLen, maxNodeNum):
    res1 = torch.zeros(maxNodeNum, dtype=torch.int32).cuda()
    res2 = torch.zeros(maxNodeNum, dtype=torch.int32).cuda()
    base_path += '/part'
    addIndex = [[] for _ in range(part_num)]
    for i in range(1, part_num + 1):
        cur_part = part_seq[i]
        next_part = part_seq[(i+1) % part_num]
        curNode = nodeList[cur_part].cuda()
        nextNode = nodeList[next_part].cuda()
        curLen = curNode.shape[0]
        nextLen = nextNode.shape[0]
        print(f"gen_add_feat,cur:{cur_part} {curLen}, next:{next_part} {nextLen}")
        
        res1.fill_(0)
        res2.fill_(0)
        dgl.findSameNode(curNode, nextNode, res1, res2)
        same_num = torch.sum(res1).item()
        
        # 索引位置 
        maxlen = max(curLen,nextLen)
        res1_zero = torch.squeeze(torch.nonzero(res1[:maxlen] == 0)).to(torch.int32)
        res2_zero = torch.squeeze(torch.nonzero(res2[:maxlen] == 0)).to(torch.int32)
        res1_one = torch.squeeze(torch.nonzero(res1[:maxlen] == 1)).to(torch.int32)
        res2_one = torch.squeeze(torch.nonzero(res2[:maxlen] == 1)).to(torch.int32)

        if (nextLen > same_num):
            if(curLen < nextLen or curLen == nextLen):
                replace_value = res2_zero.cuda()
            elif(curLen > nextLen):
                replace_value = res2_zero[:nextLen - same_num].cuda()
        else:
            replace_value = torch.Tensor([]).to(torch.int32)
            

        nextPath = base_path + str(next_part)
        sameNodeInfoPath = nextPath + '/test_sameNodeInfo.bin'
        diffNodeInfoPath = nextPath + '/test_diffNodeInfo.bin'
        sameNode = torch.cat((res1_one, res2_one), dim = 0)
        diffNode = torch.cat((res1_zero, res2_zero), dim = 0)
        saveBin(sameNode.cpu(), sameNodeInfoPath)
        saveBin(diffNode.cpu(), diffNodeInfoPath)
        sameNode, diffNode = None, None

        # 特征生成部分 TODO
        addIndex[next_part] = replace_value
        return addIndex


if __name__ == '__main__':
    # JSONPATH = "/home/bear/workspace/single-gnn/datasetInfo.json"
    JSONPATH = "/home/gr/single-gnn/datasetInfo.json"
    partitionNUM = 8
    sliceNUM = 8
    with open(JSONPATH, 'r') as file:
        data = json.load(file)
    datasetName = ["PA"] 

    for NAME in datasetName:
        GRAPHPATH = data[NAME]["rawFilePath"]
        maxID = data[NAME]["nodes"]
        subGSavePath = data[NAME]["processedPath"]
        
        # trainId = torch.tensor(np.fromfile(GRAPHPATH + "/trainIds.bin",dtype=np.int64))
        # shuffled_indices = torch.randperm(trainId.size(0))
        # trainId = trainId[shuffled_indices]
        # trainBatch = torch.chunk(trainId, partitionNUM, dim=0)
        # graph = np.fromfile(GRAPHPATH+"/graph.bin",dtype=np.int32)
        # startTime = time.time()
        # for index,trainids in enumerate(trainBatch):
        #     t = analysisG(graph,maxID,trainId=trainids,savePath=subGSavePath+f"/part{index}")
        
    #     startTime = time.time()
    #     t1 = PRgenG(GRAPHPATH,maxID,partitionNUM,savePath=subGSavePath)
    #     print(f"run time cost:{RUNTIME:.3f}")
    #     print(f"save time cost:{SAVETIME:.3f}")
    #     print(f"partition all cost:{time.time()-startTime:.3f}s")
    
        RAWDATAPATH = data[NAME]["processedPath"]
        FEATPATH = data[NAME]["rawFilePath"] + "/feat.bin"
        LABELPATH = data[NAME]["rawFilePath"] + "/labels.bin"
        SAVEPATH = data[NAME]["processedPath"]
        nodeNUM = data[NAME]["nodes"]
        featLen = data[NAME]["featLen"]
        
    #     MERGETIME = time.time()
    #     rawData2GNNData(RAWDATAPATH,partitionNUM,LABELPATH)
    #     print(f"trans graph cost time{time.time() - MERGETIME:.3f}s ...")
    
    
        diffMatrix = [[0 for _ in range(partitionNUM)] for _ in range(partitionNUM)]
        startTime1 = time.time()
        nodeList = []
        maxNodeNum,minPath = cal_min_path(diffMatrix , nodeList, partitionNUM, data[NAME]["processedPath"])
        print(f"计算最优加载路径用时{time.time() - startTime1:.4f}s")
        print(f"part最大节点数: {maxNodeNum},最优加载路径为:{res}")
        
        res = [2, 4, 3, 5, 6, 1, 0, 7]
        trainPath = np.array(res)
        saveBin(trainPath,SAVEPATH+f'/trainPath.bin')

        
        startTime1 = time.time()
        addIdx = genFeatIdx(partitionNUM, SAVEPATH, nodeList, res, featLen, maxNodeNum)
        print(f"生成addFeat用时{time.time() - startTime1:.4f}s")
        
        genAddFeat(res[0],addIdx,SAVEPATH,FEATPATH,partitionNUM,nodeNUM,sliceNUM,featLen)

    #     FEATTIME = time.time()
    #     genSubGFeat(SAVEPATH,FEATPATH,partitionNUM,nodeNUM,sliceNUM,featLen)
    #     print(f"graph feat gen cost time{time.time() - FEATTIME:.3f}...")