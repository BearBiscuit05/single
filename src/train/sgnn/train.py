import numpy as np
import time
import mmap

def mergeFeat(file_hand,sampleNodes,featLen):
    feats = np.zeros((len(sampleNodes),featLen),dtype=np.int32)
    start = time.time()   
    for index,nodeID in enumerate(sampleNodes):
        int_array_length = featLen
        int_array = np.frombuffer(file_hand, dtype=np.int32, offset=nodeID*4, count=int_array_length)
        feats[index] = int_array
    print("mmap time :", time.time()-start)
    return feats

if __name__ == "__main__":
    file_path = "./data/copy_1.bin"
    file = open(file_path, "r+b")
    mmapped_file = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)   
    featLen = 16
    feat1 = mergeFeat(mmapped_file,[1,2],featLen)

    file_path2 = "./data/copy_2.bin"
    file2 = open(file_path2, "r+b")
    mmapped_file2 = mmap.mmap(file2.fileno(), 0, access=mmap.ACCESS_READ)   
    feat2 = mergeFeat(mmapped_file2,[1,2],featLen)

    file_path3 = "./data/copy_3.bin"
    file3 = open(file_path3, "r+b")
    mmapped_file3 = mmap.mmap(file3.fileno(), 0, access=mmap.ACCESS_READ)   
    feat3 = mergeFeat(mmapped_file3,[1,2],featLen)
    print(feat1)
    print(feat2)
    print(feat3)

    mmapped_file.close()
    mmapped_file2.close()
    mmapped_file3.close()
    file.close()
    file2.close()
    file3.close()
