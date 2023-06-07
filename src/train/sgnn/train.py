import numpy as np
import time
import mmap
import sys
import os
import argparse
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
    # [0,10] 左取右不取
    sequence = np.arange(min_val, max_val)
    np.random.shuffle(sequence)
    return sequence


def env_init(part):
    randList = generate_random_sequence(0, part.part)
    print(randList)

def main():
    env_init(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='这是一个简单的命令行程序')

    parser.add_argument('--part', type=str,default=8, help='graph part number')

    args = parser.parse_args()

    main(args)