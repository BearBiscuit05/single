import numpy as np
import time
import mmap



file_path = "srcList.bin"
# 使用mmap打开文件
random_array = np.random.randint(1, 96864000, size=200000)
#random_array = np.random.randint(1, 1000, size=200000)
arrys = []
with open(file_path, "r+b") as file:
    mmapped_file = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)
    file_content = mmapped_file.read()
    start = time.time()   
    for i in range(200000):
        file_size = 32
        int_array_length = 128  # 假设每个整数占用4字节
        int_array = np.frombuffer(file_content, dtype=np.int32,offset=random_array[i], count=int_array_length)
        arrys.append(int_array)
    mmapped_file.close()
    print("mmap time :", time.time()-start)

print(len(arrys))



def mergeFeat(FileName,sampleNodes,featLen):
    feats = np.zeros((len(sampleNodes),featLen))
    with open(file_path, "r+b") as file:
        mmapped_file = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)
        file_content = mmapped_file.read()
        start = time.time()   
        for index,nodeID in enumerate(sampleNodes):
            int_array_length = featLen
            int_array = np.frombuffer(file_content, dtype=np.int32, offset=nodeID, count=int_array_length)
            feats[index] = int_array
        mmapped_file.close()
        print("mmap time :", time.time()-start)