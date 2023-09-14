import torch
import dgl
import numpy as np
import mmap


"""
--part0
----inter edges
----cut edges  --> 需要按照另外一个分区来对应id
----inter nodes
--part1
...
--result:记录了每个分区的节点数目

转换过程:
1.part-n来说,global id被修改为part-(n-1)范围的最大--part-n的范围
2.此时需要存储一个映射表-->mapping文件
3.修改cut edges中一边的id内容 --> lid
4.按照mapping对应并修改到另外一个id(要匹配到) --> nodeNUM + lid 
"""
class DataAdapter:
    def __init__(self, FilePath, PartNUM ,nodeNUM, FeatLen):
        # "/raid/bear/test_part_dataset/"
        self.raw_path = FilePath
        self.featFilePath = self.raw_path + "raw/feats.bin"
        self.nodeNUM = nodeNUM
        self.FeatLen = FeatLen
        # self.fpr = self.fetchFeatPtr()
        self.PartNUM = PartNUM
        self.GMap=torch.Tensor(nodeNUM).to(torch.int64)
        self.nodeNUMs = []


    def fetchLabel(self, labelFilePath, indices):
        label_data = np.fromfile(labelFilePath, dtype=np.int64)
        label_selected_data = label_data[indices]
        return label_selected_data

    def fetchFeat(self, indices):
        fpr = np.memmap(self.featFilePath, dtype='float32', mode='r', shape=(self.nodeNUM, self.FeatLen))
        feats = fpr[indices]
        return feats

    def loadingHalo(self, rank1, rank2):
        HaloFilePath = self.raw_path + "part" + str(rank1) + "/halo_" + str(rank2) + ".bin"
        HaloList = self.bin2tensor(HaloFilePath, datatype=np.int64)
        return HaloList
    
    def loadingPartNode(self, rank):
        NodeFilePath = self.raw_path + "part" + str(rank) + "/NodeIds.bin"
        NodeList = self.bin2tensor(NodeFilePath, datatype=np.int64)
        return NodeList

    def TransEngine(self):
        # 1.构建映射表，顺序加载图的node和edge
        # 2.构建全局map,调整局部id
        sumNodeNUM = 0
        for rank in range(self.PartNUM):
            partPath = self.raw_path + "part" + str(rank) + "/"
            nodeids=self.loadingPartNode(rank)
            nodeNUM = len(nodeids)
            self.nodeNUMs.append(nodeNUM)
            
            # 存储newIndex
            newIndex = torch.arange(nodeNUM) + sumNodeNUM
            newIndexPath = partPath+"new_nodeId.bin"
            self.tensor2bin(newIndex,newIndexPath)
            sumNodeNUM += nodeNUM
            
            self.GMap[nodeids]=newIndex

            # 抽取存储特征
            feats=self.fetchFeat(nodeids)
            featPath=partPath+"feats.bin"
            self.tensor2bin(feats,featPath)

            # 抽取指定训练标签

        print("creat GMap success...")

        # 3.调整分割边
        for rank1 in range(self.PartNUM):
            for rank2 in range(self.PartNUM):
                if rank1 == rank2:
                    continue
                HaloList = self.loadingHalo(rank1,rank2)
                HaloList = self.GMap[HaloList]
                haloPath="/raid/bear/test_part_dataset/part"+str(rank1)+"/new_halo"+str(rank2)+".bin"
                self.tensor2bin(HaloList,haloPath)
        

    def bin2tensor(self, filePath, datatype=np.int64):
        tensor = np.fromfile(filePath, dtype=datatype)
        return tensor
    
    def tensor2bin(self, variable, fileSavePath):
        if torch.is_tensor(variable):
            variable = variable.numpy()
        elif isinstance(variable, np.ndarray):
            pass
        else:
            raise ValueError("Input variable must be a PyTorch tensor or a NumPy array")     
        variable.tofile(fileSavePath)

if __name__ == '__main__':
    raw_path="/raid/bear/test_part_dataset/"
    adapter = DataAdapter(raw_path,4,200,100)
    adapter.TransEngine()