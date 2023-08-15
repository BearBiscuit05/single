import torch
import signn
import time

def transmapping(nodeList):
    mapping_tensor = torch.unique(nodeList)
    # print(mapping_tensor)
    # print(len(mapping_tensor))
    # print(nodeList)
    signn.torch_graph_mapping(nodeList,mapping_tensor,len(nodeList),len(mapping_tensor))
    #print(nodeList)
    maxNUM,_ = torch.max(nodeList,dim=0)
    minNUM,_ = torch.min(nodeList,dim=0)
    #print("maxNUM:",maxNUM,"  minNUM:",minNUM)

if __name__ == "__main__":
    num_elements = 256000
    min_value = 0
    max_value = 2000000  # 最大的 int32 值
    random_int_tensor = torch.randint(min_value, max_value + 1, (num_elements,), dtype=torch.int32).to('cuda:0')
    t = time.time()
    for i in range(300):
        transmapping(random_int_tensor)
    print("computeTime :",time.time()-t)