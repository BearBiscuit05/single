import time
import copy
import torch
ll=[i for i in range(256000)]
l=torch.tensor(ll)
z = torch.Tensor([0])
l_z = torch.cat([z,l])

start=time.time()
l=l.to('cuda:0')
print(time.time()-start)
l=l.cpu()
start=time.time()
l=l.to('cuda:0')
print(time.time()-start)

tmp=torch.tensor(ll)
tmp=tmp.to('cuda:0')
tmp=tmp.cpu()

start=time.time()
z = torch.Tensor([0])
print(time.time()-start)

start=time.time()
tmp1 = torch.cat([torch.Tensor([0]),tmp])
print(time.time()-start)

start=time.time()
l_z[1:]=tmp
print(time.time()-start)
