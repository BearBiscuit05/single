import numpy as np
import torch

def count_parameters(model):
    return sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad])


if __name__ == '__main__':
    sage = ACC_SAGE(128, 512, 172, 3, torch.nn.functional.relu, 0)
    print(count_parameters(sage))