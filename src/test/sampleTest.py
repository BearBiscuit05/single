import numpy as np

def sampleNeig(sampleIDs,cacheGraph,sampleBound):
    # 需要返回一个多维list，与fanout的长度一致
    # 数据准备部分
    fanout = [3,3]
    cacheData = []
    cacheData.append([i for i in range(100000)])
    cacheData.append([i*10 for i in range(10000)])

    #sampleList = sampleIDs[sampleBound[0]:sampleBound[1]]
    layer = len(fanout)
    bound = [sampleBound[0],sampleBound[1]]
    sampleIndex = [i for i in range(sampleBound[0],sampleBound[1])]
    
    for l, number in enumerate(fanout):
        number -= 1
        if l != 0:     
            last_lens = len(cacheGraph[layer-l][0])      
            cacheGraph[layer-l-1][0][0:last_lens] = cacheGraph[layer-l][0]
            cacheGraph[layer-l-1][1][0:last_lens] = cacheGraph[layer-l][0]
        else:
            last_lens = len(sampleIDs)
            cacheGraph[layer-l-1][0][0:last_lens] = sampleIDs
            cacheGraph[layer-l-1][1][0:last_lens] = sampleIDs
        print("bound:{}".format(bound))
        for index in sampleIndex:
            ids = cacheGraph[layer-l-1][0][index]
            if ids == -1:
                continue
            NeigList = cacheData[0][cacheData[1][ids*2]+1:cacheData[1][ids*2+1]]
            if len(NeigList) < number:
                sampled_values = NeigList
            else:
                sampled_values = np.random.choice(NeigList,number)
            
            offset = last_lens + (index * number)
            fillsize = len(sampled_values)
            cacheGraph[layer-l-1][0][offset:offset+fillsize] = sampled_values # src
            cacheGraph[layer-l-1][1][offset:offset+fillsize] = [ids] * fillsize # dst
        
        #sampleList = cacheGraph[layer-l-1][0][bound[0]*number:bound[1]*number]
        bound = [last_lens+bound[0]*number,last_lens+bound[1]*number]
        tmp = [i for i in range(bound[0],bound[1])]
        sampleIndex.extend(tmp)

cacheGraph = []
fanout = [3,3]
sampleIDs = [2,4,6,7]
tmp = len(sampleIDs)
number = len(sampleIDs)
for layer, fan in enumerate(fanout):
    dst = [-1] * tmp * fan
    src = [-1] * tmp * fan
    cacheGraph.insert(0,[src,dst])
    tmp = tmp * fan
    number += tmp
print(cacheGraph)
sampleNeig(sampleIDs,cacheGraph,[0,4])
print(cacheGraph)