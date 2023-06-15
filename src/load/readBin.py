import numpy as np
import time
import mmap

def mergeFeat(mmap_file_hand,sampleNodes,featLen):
    feats = np.zeros((len(sampleNodes),featLen),dtype=float)
    int32_size = np.dtype(float).itemsize
    start = time.time()   
    for index,nodeID in enumerate(sampleNodes):
        int_array_length = featLen
        offset = nodeID *featLen* int32_size
        int_array = np.frombuffer(mmap_file_hand, dtype=float, offset=offset, count=int_array_length)
        feats[index] = int_array
    print("mmap time :", time.time()-start)
    return feats

if __name__ == "__main__":
    sampleNodes = [i for i in range(10)]
    featLen = 20
    file_path = "../../data/products/part0/feat.bin"
    file = open(file_path, "r+b")
    mmapped_file = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)
    feat = mergeFeat(mmapped_file,sampleNodes,featLen)
    print(feat)
    mmapped_file.close()
    file.close()