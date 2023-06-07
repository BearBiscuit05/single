import numpy as np
import time
import mmap
import sys
import os
current_folder = os.path.abspath(os.path.dirname(__file__))
#sys.path.append(current_folder+"/../"+"utils")


def mergeFeat(file_hand,sampleNodes,featLen):
    feats = np.zeros((len(sampleNodes),featLen),dtype=np.int32)
    start = time.time()   
    for index,nodeID in enumerate(sampleNodes):
        int_array_length = featLen
        int_array = np.frombuffer(file_hand, dtype=np.int32, offset=nodeID*4, count=int_array_length)
        feats[index] = int_array
    print("mmap time :", time.time()-start)
    return feats

def generate_random_sequence(min_val, max_val):
    sequence = np.arange(min_val, max_val)
    np.random.shuffle(sequence)
    return sequence


def env_init():
    pass

def main():
    pass

if __name__ == "__main__":
    main()
    print(current_folder)