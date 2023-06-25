import array
import torch
import mmap
import numpy as np
a = torch.tensor([[0.,1.,2.,3.,4.,5.,6.,7.,8.,9.],[10.,11.,12.,13.,14.,15.,16.,17.,18.,19.]],dtype=torch.float32)
a = a.detach().numpy() 
a.tofile("tmp.bin")
file_path = "./tmp.bin"
file = open(file_path, "r+b")
mmapped_file = mmap.mmap(file.fileno(), 0)
int_size = np.dtype(np.float32).itemsize
t = np.frombuffer(mmapped_file, dtype=np.float32, offset=1 * int_size, count=2)
t1 = torch.frombuffer(mmapped_file, dtype=torch.float32,offset=3 * int_size, count=2)
# t1 = t1.to(torch.float32)
print("numpy: {}".format(t))
print("torch : {}".format(t1))
print(t1.dtype)
del t
del t1
mmapped_file.close()
file.close()


