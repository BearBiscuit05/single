import torch
import signn
import time
import copy
def transmapping(nodeList):
    raw = copy.deepcopy(nodeList)
    mapping_tensor = torch.unique(nodeList)
    signn.torch_graph_mapping(nodeList,mapping_tensor,len(nodeList),len(mapping_tensor))
    for i in range(len(nodeList)):
        if(mapping_tensor[nodeList[i]] != raw[i]):
            print("error...")
            exit()

if __name__ == "__main__":
    num_elements = 256000
    min_value = 0
    max_value = 2000000  # 最大的 int32 值
    random_int_tensor = torch.randint(min_value, max_value + 1, (num_elements,), dtype=torch.int32).to('cuda:0')
    transmapping(random_int_tensor)