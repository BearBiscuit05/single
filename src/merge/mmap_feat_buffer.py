import numpy as np
import time
import mmap


def mergeFeat(mmap_file_hand,sampleNodes,featLen):
    feats = np.zeros((len(sampleNodes),featLen),dtype=np.int32)
    int32_size = np.dtype(np.int32).itemsize
    start = time.time()   
    for index,nodeID in enumerate(sampleNodes):
        int_array_length = featLen
        int_array = np.frombuffer(mmap_file_hand, dtype=np.int32, offset=nodeID * int32_size, count=int_array_length)
        feats[index] = int_array
    print("mmap time :", time.time()-start)
    return feats

if __name__ == "__main__":
    file_path = "./data/srcList.bin"
    file = open(file_path, "r+b")
    mmapped_file = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)
    int_array = np.frombuffer(mmapped_file, dtype=np.int32, offset=0, count=8)    
    random_array = np.random.randint(1, 10, size=2)
    featLen = 16
    feats = mergeFeat(mmapped_file,random_array,featLen)
    print(int_array)
    mmapped_file.close()
    file.close()
