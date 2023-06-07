import numpy as np
import time
import mmap
import sys
import os
import argparse
import sample_hop
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

def initFeatFile(dataset,number):
    pass

def loadGraph(dataset,partNUM):
    graphPath = "{}/../../data/{}/part{}".format(current_folder, dataset,partNUM)
    srcPath = graphPath + "srcList.bin"
    boundPath = graphPath + "bound.bin"
    srcList = np.fromfile(srcPath, dtype=np.int32)
    boundList = np.fromfile(boundPath, dtype=np.int32)
    return srcList,boundList

def train():
    pass

def sampleSubG(srcs, bound,args):
    numbers = [int(num) for num in args.fanouts.split(",")]
    sample_hop.torch_launch_sample_2hop()


def run(args):
    randList = generate_random_sequence(0, args.part)
    for grapgID in randList:
        srcs, bound = loadGraph(args.dataset,grapgID)
        # TODO: sample
        sample_hop.torch_launch_sample_2hop()
        # TODO: merge
        train()

def main(args):
    run(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='这是一个简单的命令行程序')

    parser.add_argument('--part', type=int,default=8, help='graph part number')
    parser.add_argument('--epoch', type=int,default=10, help='train epoch')
    parser.add_argument('--dataset', type=str,default='ogbn_products', help='dataset name')
    parser.add_argument('--fanouts', type=str,default='10,25', help='sample fanouts, hop1,hop2...')
    args = parser.parse_args()
    main(args)
