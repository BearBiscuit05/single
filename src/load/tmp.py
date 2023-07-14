import numpy as np

def save_edges_bin(nodeDict, filepath, haloID, nodeNUM, edgeNUM):
    edges = []
    bound = [0]
    ptr = 0
    for key in range(nodeNUM):
        if key in nodeDict:
            srcs = nodeDict[key]
            for srcid in srcs:
                edges.extend([srcid,key])
            ptr += len(srcs)*2
            bound.append(ptr)
        else:
            bound.append(ptr)
    edges = np.array(edges,dtype=np.int32)
    bound = np.array(bound,dtype=np.int32)
    print(edges)
    print(bound)
    #edges.tofile(filepath+"/halo"+str(haloID)+".bin")
    #bound.tofile(filepath+"/halo"+str(haloID)+"_bound.bin")

if __name__ == '__main__':
    nodeDict = {1:[11,12,13,14],2:[23,22,28],4:[42,45,46,47,49]}
    print(nodeDict[1])
    save_edges_bin(nodeDict, '.', 0, 5, 1)