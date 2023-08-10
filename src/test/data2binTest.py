"""
测试图数据转换为二进制文件时不产生错误:
# 测试流程及进度
## 数据转换测试
dgl->bin file
1.测试图数据从dgl数据集(边集)转换到srclist与bound时是否有问题
2.测试特征的转换是否存在问题
3.测试数据标签的转换是否存在问题(已完成)
4.测试训练集，验证集，测试集是否有问题(已完成)
5.测试halo部分的输出是否可以还原至原来
期望是可以使得我们的二进制数据可以从新加载到dgl中,与dgl原本的图+特征完全一致
"""
import numpy as np
import dgl
import torch

class DGL2BinTester(object):
	def __init__(self,datasetName,rawDataPath,newDataPath,partNum):
		self.datasetName = datasetName
		self.rawDataPath = rawDataPath
		self.newDataPath = newDataPath
		self.partNum = partNum
		self.partConfigJson = rawDataPath + datasetName + '.json'
		print("DGL2BinTester has been initialized:\ndataset:{}\nraw path:{}\nbin path:{}\npart num:{}\njson:{}\n".format(self.datasetName,self.rawDataPath,self.newDataPath,self.partNum,self.partConfigJson))

	def loadDGLData(self):
		print("start to load raw data from dgl...")
		datas = []
		for partIndex in range(self.partNum):
			subg, node_feat, _, _, _, node_type, _ = dgl.distributed.load_partition(self.partConfigJson,partIndex)
			data = (subg,node_feat,node_type)
			datas.append(data)
		return datas
	
	def testID(self,datas):
		# datas: loadDGLData返回得到的各个分区图数据的列表
		print("start to test IDs....")
		flag = True
		for partIndex in range(self.partNum):
			# 加载原始分区数据
			subg,node_feat,node_type = datas[partIndex]
			# 取出特征tensor转换成numpy
			trainIDRawData = node_feat[node_type[0]+'/train_mask'].nonzero().squeeze()
			valIDRawData = node_feat[node_type[0]+'/val_mask'].nonzero().squeeze()
			testIDRawData = node_feat[node_type[0]+'/test_mask'].nonzero().squeeze()
			# 加载转换后的二进制数据到numpy
			trainIDBinFile = self.newDataPath + 'part' + str(partIndex) + '/trainID.bin'
			trainIDBinData = torch.load(trainIDBinFile).to(torch.uint8).nonzero().squeeze()
			valIDBinFile = self.newDataPath + 'part' + str(partIndex) + '/valID.bin'
			valIDBinData = torch.load(valIDBinFile).to(torch.uint8).nonzero().squeeze()
			testIDBinFile = self.newDataPath + 'part' + str(partIndex) + '/testID.bin'
			testIDBinData = torch.load(testIDBinFile).to(torch.uint8).nonzero().squeeze()
			# 两个int32的numpy数组做比较，一致说明转换正确
			if trainIDRawData.equal(trainIDBinData) == False:
				print('error in part %d train id'%partIndex)
				flag = False
			if valIDRawData.equal(valIDBinData) == False:
				print('error in part %d val id'%partIndex)
				flag = False
			if testIDRawData.equal(testIDBinData) == False:
				print('error in part %d test id'%partIndex)
				flag = False
		if flag == True:
			print('0 errors in (train/val/test)ids in all parts.')
		return flag
	
	def testLabel(self,datas):
		# datas: loadDGLData返回得到的各个分区图数据的列表
		print("start to test Label....")
		flag = True
		for partIndex in range(self.partNum):
			# 加载原始分区数据
			subg,node_feat,node_type = datas[partIndex]
			# 取出特征tensor转换成numpy
			labelRawData = node_feat[node_type[0]+'/labels'].numpy()
			# 加载转换后的二进制数据到numpy
			labelBinFile = self.newDataPath + 'part' + str(partIndex) + '/label.bin'
			labelBinData = np.fromfile(labelBinFile, dtype=np.int32)
			# 两个int32的numpy数组做比较，一致说明转换正确
			comparison = labelRawData == labelBinData
			if comparison.all() == False:
				print('error in part %d labels'%partIndex)
				flag = False
		if flag == True:
			print('0 errors in labels in all parts.')
		return flag
	
	def testGraph(self):
		pass
	
	def testFeat(self):
		pass
	
	def main_test(self):
		falg = True
		datas = self.loadDGLData()
		if self.testLabel(datas) == False:
			flag = False
		if self.testID(datas) == False:
			flag = False
		return flag

if __name__ == '__main__':
	datasetName = 'ogb-product'
	rawDataPath = '/home/bear/workspace/singleGNN/data/raw-products_4/'
	newDataPath = '/home/bear/workspace/singleGNN/data/products_4/'
	partNum     = 4
	tester      = DGL2BinTester(datasetName,rawDataPath,newDataPath,partNum)
	if tester.main_test() == False:
		print('errors exist!')
	else:
		print('pass all checks!')