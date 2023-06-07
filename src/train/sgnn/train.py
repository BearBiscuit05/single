import numpy as np
import time
import struct
import gc
import mmap

if __name__ == "__main__":
    # file_path = './data/copy_1.bin'
    # start = time.time()
    # graphEdge1 = np.fromfile(file_path, dtype=np.int32)
    # print(time.time()-start)
    # print(graphEdge1)

    # file_path = './data/copy_2.bin'
    # start = time.time()
    # graphEdge2 = np.fromfile(file_path, dtype=np.int32)
    # print(time.time()-start)
    # print(graphEdge2)
    # del graphEdge1
    # file_path = './data/copy_3.bin'
    # start = time.time()
    # graphEdge3 = np.fromfile(file_path, dtype=np.int32)
    # print(time.time()-start)
    # print(graphEdge3)


    file_path = "./data/copy_2.bin"
    random_array = np.random.randint(1, 96864000, size=200000)
    arrys = []
    with open(file_path, "r+b") as file:
        start = time.time()
        mmapped_file = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)
        file_content = mmapped_file.read()
           
        for i in range(200000):
            file_size = 32
            int_array_length = 128  # 假设每个整数占用4字节
            int_array = np.frombuffer(file_content, dtype=np.int32,offset=random_array[i], count=int_array_length)
            arrys.append(int_array)
        mmapped_file.close()
        print("mmap time :", time.time()-start)
    print(len(arrys))