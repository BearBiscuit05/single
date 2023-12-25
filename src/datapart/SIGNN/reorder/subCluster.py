import numpy as np
import os
import torch
import dgl
import gc
import time
import torch
import math

# 以上过程均在CUDA中完成

subGNUM = []    # 
subGCost = []   #subGCost[i]表示子图i的总开销
subGMap = []    #subGMap[i]表示当前子图i的实际子图位置(用于合并操作)
subGWeight = [] #weight表示这个分区是由多少个初始子分区合并而来的
subGTrack = []  #subGTrack[i] = [] 表示分区i由哪些初始分区构成的

mergeBound = 0
averDegree = 0
featLen = 0

def findMinPart(partInfo):
    min_value, min_index = torch.min(partInfo, dim=0)
    return min_value, min_index

def findMaxPart(partInfo):
    max_value, max_index = torch.max(partInfo, dim=0)
    return max_value, max_index

def changeInfo(partInfo,changeIdx,changeValue,acc=True):
    if acc:
        partInfo[changeIdx] += changeValue
    else:
        partInfo[changeIdx] = changeValue

def genSmallCluster(trainids,nodeTable,partNUM):
    torch.cuda.empty_cache()
    gc.collect()
    # 提取训练集标签
    trainLabel = nodeTable[trainids.to(torch.int64)]

    # 按照标签进行聚类(数量不定)
    labelCluster = torch.zeros(torch.max(trainLabel).item()+1,dtype=torch.int32,device="cuda")
    dgl.bincount(trainLabel,labelCluster)
    
    startTime = time.time()
    sortLabelsNUM,labelIdx = torch.sort(labelCluster,descending=True)

    bound = len(torch.nonzero(sortLabelsNUM > 0).reshape(-1))
    sortLabelsNUM = sortLabelsNUM[:bound]
    labelIdx = labelIdx[:bound]
    
    # 将小聚类进行合并，生成为具体的聚类数目
    partInfo = torch.zeros(partNUM,dtype=torch.int64,device="cuda")
    label2part = torch.zeros(torch.max(labelIdx).item()+1,dtype=torch.int64)

    #直接将labelIdx打乱，然后partNUM等分
    shuffle_idx = torch.randperm(labelIdx.shape[0])
    labelIdx = labelIdx[shuffle_idx]
    left = 0
    bound = right = labelIdx.shape[0] // partNUM + 1
    for i in range(partNUM):
        label2part[labelIdx[left:right].to(torch.int64)] = i
        left = right
        right = min(labelIdx.shape[0], right + bound)

    print(f"mergeTime time :{time.time()-startTime:.2f}s")
    startTime = time.time()
    # 此时所有的小聚类均放入大聚类中，id: 0 - (partNUM-1)
    trainIdsInPart = label2part[trainLabel.to(torch.int64)] # 最终获得每个训练点在哪个分区中
    for i in range(partNUM):
        print(f"{i}:{torch.nonzero(trainIdsInPart == i).shape[0]}", end = "|")
    return trainIdsInPart
    # 最后在训练时对nodeInfo[trainids] = trainIdsInPart

def countLabelNUM(nodeTable,label):
    #计算对应的Label有多少值,并同步subGNUM
    nodeIndex = (nodeTable & (1 << label)) != 0
    NUM = nodeIndex.sum()
    subGNUM[label] = NUM
    return NUM

def findLabelNodes(nodeTable,label):
    # 查询对应Label的顶点
    nodeIndex = (nodeTable & (1 << label)) != 0
    NodeIds = torch.nonzero(nodeIndex).reshape(-1)
    return NodeIds.to(torch.int32).cuda()


def mergeLabelCost(nodeTable,label_1,label_2):
    # 计算合并两个标签需要多少开销
    Label1Nodes = findLabelNodes(nodeTable,label_1)
    Label2Nodes = findLabelNodes(nodeTable,label_2)
    indexTable1 = torch.zeros_like(Label1Nodes,dtype=torch.int32,device="cuda")
    indexTable2 = torch.zeros_like(Label2Nodes,dtype=torch.int32,device="cuda")
    dgl.findSameNode(Label1Nodes,Label2Nodes,indexTable1,indexTable2)   # 返回的是0/1
    sameNum = torch.sum(indexTable1).item()
    diffNum = (Label1Nodes.shape[0] + Label2Nodes.shape[0] - sameNum)
    # 这里需要考虑两点 1.每个点的平均出度 2.每个点的特征带来的大小
    #需要保证图结构不会超过用户指定的显存    +
    #同时应当保证总体上的所有分区都可以塞到显存里
    return (diffNum * averDegree + diffNum * featLen) / 1024 / 1024 * 4


def strategy_single(nodeTable, maxBound):
    #选择相同层次的，cost最大和最小的两个的两个分区进行合并，当任一分区超过maxBound或是层次超过mergeBound时推出
    for i in range(int(math.log2(mergeBound))):
        while True:
            validIdx = torch.nonzero(subGWeight == i + 1).reshape(-1)
            if (validIdx.shape[0] < 2):
                break
            costs = subGCost[validIdx]
            costs,idx = torch.sort(costs)
            idx = torch.tensor([idx[0],idx[-1]])
            if (idx[0] == idx[-1]):
                continue
            validIdx = validIdx[idx]#取最大和最小的两个
            validIdx,_ = torch.sort(validIdx) #按索引排序，因为后面merge的时候固定是合并成小标签
            curCost = mergeLabelCost(nodeTable,validIdx[0],validIdx[1])
            
            if (curCost > maxBound):
                subGWeight[validIdx] += 1#将这两个节点放到下一个层级，即跳过这两个节点
                break
            
            #此时执行合并操作,维护公共数据

            mergeLabel(nodeTable, validIdx[0],validIdx[1])
            subGWeight[validIdx[0]] *= 2 #由两个同层次的合并而来
            subGWeight[validIdx[1]] = -1
            subGCost[validIdx[0]] = curCost
            subGCost[validIdx[1]] = -1


def mergeMain(nodeTable, maxBound, strategyName):
    
    # 寻找可以合并的两个Label
    #1.随机合并，随机挑选两个同量级的分区进行合并，若开销不可承受，则重新随机。（随机的顺序应当提前既定，防止重复的随机数出现）
    #2.贪心合并，挑选两个同量级的分区进行合并，要求本次合并的开销是局部最优的
    #3.穷举合并，穷举所有方式，得出全局最优的方式

    #同时需要注意，是优先选择同量级的分区进行合并使得这个过程为32 - 16 - 8
    #还是优先选择可以合并的分区进行合并，过程可能变成32 - 31 - 30 - 28 - ...。但这种方法可能会导致最后的分区大小不均衡

    #这里实现方法：简单贪心合并并且优先选择同量级分区
    strategy_func = globals()["strategy_" + strategyName]
    result = strategy_func(nodeTable, maxBound)
    return result

def mergeLabel(nodeTable,label_1,label_2):
    value_1 = ((nodeTable >> label_1) & 1) != 0 # 是否在label_1聚类中
    value_2 = ((nodeTable >> label_2) & 1) != 0 # 是否在label_2聚类中

    mergeNodes = value_1 | value_2

    max_position = max(label_1, label_2)
    min_position = min(label_1, label_2)

    # 将大值对应的位置置为0
    newTable = nodeTable & ~(1 << max_position) # 大标签归0
    newTable = newTable | (1 << min_position)   # 小标签置1

    nodeTable[mergeNodes] = newTable[mergeNodes]    # 只有至少一个位置为1时才执行操作

    subGNUM[label_2] = 0
    subGNUM[label_1] = countLabelNUM(nodeTable, label_1)
    subGTrack[label_1] += subGTrack[label_2]

"""
tensor = torch.tensor([1, 5, 10, 15, 4], dtype=torch.int32)
result = custom_operation(tensor, 0, 2)
"""

#maxBound单位应当是MB
#mergeBound: 合并阈值，即至多支持多少个原始分区合并。例如初始32分区，mergeBount = 4，就是最多支持4个初始分区合并，最后的结果一般就是32 / 4 = 8个子图分区
def startCluster(nodeTable,cluserNUM,maxBound,globalData):

    # nodeTable最开始的标签数目暂时定为32，之后要合并到一个值，按照我的理解，最好是32->16->8然后一般的不能到4了，因为会爆
    # 不过UK倒是可以，总之就是这样的
    
    # trans.py的调用函数
    global subGNUM,subGWeight,subGMap,subGCost,subGTrack
    global mergeBound,averDegree,featLen
    subGCost = []
    subGMap = []
    subGTrack = []

    mergeBound,averDegree,featLen = globalData
    subGNUM = torch.tensor([ 0 for _ in range(cluserNUM)],dtype=torch.int32)
    subGWeight = torch.tensor([ 1 for _ in range(cluserNUM)],dtype=torch.int32)
    #初始化子图cost。cost = (nodeNUM * averDegree * 4 + nodeNUM * 4 * featLen) / 1024 / 1024 (单位MB)
    for i in range(cluserNUM):
        nodeNUM = findLabelNodes(nodeTable, i).shape[0]
        cost = (nodeNUM * averDegree + nodeNUM * featLen) / 1024 / 1024 * 4
        subGCost.append(cost)
        subGTrack.append([i])

    subGCost = torch.tensor(subGCost, dtype = torch.float32).reshape(-1)
    subGMap = torch.arange(cluserNUM, dtype = torch.int32).reshape(-1)

    # 计算初始各个分区大小
    for labelId in range(cluserNUM):
        countLabelNUM(nodeTable,labelId)
    
    mergeMain(nodeTable, maxBound, "single")
    subGMapRes = torch.nonzero(subGNUM != 0).reshape(-1)
    print(f"最终分区结果: {subGNUM[torch.nonzero(subGNUM != 0).reshape(-1)]}\n各分区的cost: {subGCost[torch.nonzero(subGNUM != 0).reshape(-1)]}")
    # 返回最后合并的结果
    # 返回实际分区位置
    # 返回trainBatch
    return nodeTable,subGMapRes,subGTrack

#Test
# cluserNUM = 32
# nodeTable = torch.from_numpy(np.fromfile('./nodeTable.bin',dtype = np.int32))
# maxBound = 99999999999
# mergeBound = 4
# averDegree = 51
# featLen = 100
# globalData = (mergeBound, averDegree, featLen)
# startCluster(nodeTable, cluserNUM, maxBound, globalData)

def transPartId2Bit(trainIdsInPart,trainIds,nodeNUM,TableNUM,labelTableLen):
    MultNodeLabels = []
    for i in range(labelTableLen):
        MultNodeLabels.append(torch.zeros(nodeNUM,dtype=torch.int32,device="cuda"))

    #labelTableLen视为不同维度
    #0 - 29, 30 - 59, 60 - 99...
    for i in range(labelTableLen):
        curMask = (trainIdsInPart > (i * TableNUM - 1)) & (trainIdsInPart < ((i + 1) * TableNUM))
        curTrainIdsInPartIdx = torch.nonzero(curMask).reshape(-1).to(torch.int64)
        MultNodeLabels[i][trainIds[curTrainIdsInPartIdx]] = (1 << (trainIdsInPart[curTrainIdsInPartIdx] % TableNUM)).to(torch.int32).cuda()
    nodeInfo = torch.stack(tuple(MultNodeLabels),dim = 1).reshape(-1).cpu()
    return nodeInfo