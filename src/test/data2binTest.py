"""
测试图数据转换为二进制文件时不产生错误:
# 测试流程及进度
## 数据转换测试
dgl->bin file
1.测试图数据从dgl数据集(边集)转换到srclist与bound时是否有问题(已完成)
2.测试特征的转换是否存在问题(已完成)
3.测试数据标签的转换是否存在问题(已完成)
4.测试训练集，验证集，测试集是否有问题(已完成)
5.测试halo部分的输出是否可以还原至原来(已完成)
期望是可以使得我们的二进制数据可以从新加载到dgl中,与dgl原本的图+特征完全一致
"""
import numpy as np
import dgl
import torch
import json

class DGL2BinTester(object):
	def __init__(self,datasetName,rawDataPath,newDataPath,partNum):
		self.datasetName = datasetName
		self.rawDataPath = rawDataPath
		self.newDataPath = newDataPath
		self.partNum = partNum
		self.partConfigJson = rawDataPath + datasetName + '.json'
		print("DGL2BinTester has been initialized:\ndataset:{}\nraw path:{}\nbin path:{}\npart num:{}\njson:{}\n".format(self.datasetName,self.rawDataPath,self.newDataPath,self.partNum,self.partConfigJson))

	def loadDGLData(self):
		# 使用DGL加载转换前分区图的原始数据
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
	
	def testGraph(self,datas):
		# datas: loadDGLData返回得到的各个分区图数据的列表
		print("start to test Graph....")
		flag = True

		with open(self.partConfigJson,'r') as f:
			SUBGconf = json.load(f)
		# 使用读取的数据
		boundRange = SUBGconf['node_map']['_N']

		for partIndex in range(self.partNum):
			print('test part ',partIndex)
			# 加载转换后二进制数据，包括srcList.bin+range.bin(子图内)和halo.bin+halo_bound.bin(跨子图)
			srcListBinFile = self.newDataPath + 'part' + str(partIndex) + '/srcList.bin'
			srcListBinData = np.fromfile(srcListBinFile,dtype=np.int32).tolist()
			rangeBinFile = self.newDataPath + 'part' + str(partIndex) + '/range.bin'
			rangeBinData = np.fromfile(rangeBinFile,dtype=np.int32).tolist()

			halo = []
			halo_bound = []
			for p in range(self.partNum):
				halopFile = self.newDataPath + 'part' + str(partIndex) + '/halo' + str(p) + '.bin'
				halopData = np.fromfile(halopFile,dtype=np.int32).tolist()
				halop_boundFile = self.newDataPath + 'part' + str(partIndex) + '/halo' + str(p) + '_bound.bin'
				halop_boundData = np.fromfile(halop_boundFile,dtype=np.int32).tolist()
				halo.append(halopData)
				halo_bound.append(halop_boundData)
			
			# 先存入二进制数据中srcList的部分
			binsrcs = []
			for j in range(len(rangeBinData)//2):
				l,r = rangeBinData[j*2],rangeBinData[j*2+1]
				if r-l == 1 and srcListBinData[l] == j:
					binsrcs.append([])
				else:
					binsrcs.append(srcListBinData[l:r])
			
			# 加载原始分区数据
			subg,node_feat,node_type = datas[partIndex]
			src = subg.edges()[0].tolist()
			dst = subg.edges()[1].tolist()
			
			# 此处测试发现：srcList.bin中一些结点的自循环边存在重复
			# 不是predata的问题，而是dgl原始图数据中，这些自循环边也重复了
			# edgeDict = {}
			# for index in range(len(src)):
			# 	srcid,dstid = src[index],dst[index]
			# 	key = str(srcid+boundRange[partIndex][0]) + '->' + str(dstid+boundRange[partIndex][0])
			# 	if key not in edgeDict:
			# 		edgeDict[key] = 1
			# 	else:
			# 		edgeDict[key] = edgeDict[key] + 1
			# for key in edgeDict:
			# 	if edgeDict[key] > 1:
			# 		print('repeat:',key,edgeDict[key])
			inner = subg.ndata['inner_node'].tolist()
			innernode = subg.ndata['inner_node'].sum()
			nodeDict = {}
			partdict = []
			for i in range(self.partNum):
				partdict.append({})
			basiclen = SUBGconf['node_map']['_N'][partIndex][1] - SUBGconf['node_map']['_N'][partIndex][0]
			incount = 0
			outcount = [0 for i in range(self.partNum)]
			
			# 第一阶段，正向验证，原始数据里的edge都可以在转换后的二进制数据中找到，不管src是否在子图内部
			print('[stage 1]:check if all edges in raw data can be found in .bin file')
			for index in range(len(src)):
				srcid,dstid = src[index],dst[index]
				# 两点都在原始数据的子图内的点，检测其是否写入了srcList.bin
				if inner[srcid] == 1 and inner[dstid] == 1:
					if srcid not in binsrcs[dstid]:
						flag = False
						print('error! part%d edge(%d->%d) not exist in srcList.bin'%(partIndex,srcid,dstid))
				# dstid在子图内的点，检测其srcid是否写入了对应的halo
				# 检查在partid子图中的srcid到dstid的这条边是否写入对应的halo.bin和halo_bound.bin
				elif inner[srcid] != 1 and inner[dstid] == 1:
					local_srcid = srcid
					srcid = subg.ndata[dgl.NID][srcid] # srcid ：local 查询全局ID
					partid=-1
					for pid,(left,right) in enumerate(boundRange):
						if left <= srcid and srcid < right:
							partid = pid
							break
					if dstid not in partdict[partid]:
						partdict[partid][dstid] = []
					boundl,boundr = halo_bound[partid][dstid],halo_bound[partid][dstid+1]
					newsrcid = srcid - SUBGconf['node_map']['_N'][partid][0] + basiclen # 写入的id，是2个子图拼在一起的偏移id
					partdict[partid][dstid].append(newsrcid)
					
					found_flag = False
					# 在左右边界中查询
					for ss in range(boundl,boundr,2):
						mysrcid = halo[partid][ss]
						mydstid = halo[partid][ss+1]
						if mysrcid == newsrcid:
							found_flag = True
						if mydstid != dstid:# 边界范围内的dst应该对应，顺便检查halo和halo_bound
							print('error in part%d/halo%d.bin with dst = %d'%(partIndex,partid,dstid))
					if found_flag == False:
						flag = False
						print('error! In part%d graph,edge(%d->%d) exists in dgl raw data but not in halo%d.bin'%(partIndex,newsrcid,dstid,partid))	

			# 第二阶段，反向验证，二进制文件里的edge，都可以在原始数据里查询到，此部分GPU内操作
			cuda_subg = subg.to('cuda:0')
			print('[stage 2]:check if all edges in .bin file can be found in raw data')
			# 先验证srcList.bin的二进制数据能不能在原始dgl数据中找到
			for bindstid in range(len(binsrcs)):
				if len(binsrcs[bindstid]) == 0:
					continue
				binsrcids = torch.tensor(binsrcs[bindstid],device='cuda:0')
				bindstids = torch.tensor([bindstid for i in range(len(binsrcs[bindstid]))],device='cuda:0')
				# dgl子图内部边检测函数，一次传入n条边
				comp = cuda_subg.has_edges_between(binsrcids,bindstids)
				if comp.all() == False:
					false_indices = torch.nonzero(~comp)
					for fi in false_indices:
						false_srcid = binsrcids[fi]
						if false_srcid != bindstid:
							print('error!There is an edge(%d->%d) exists in srcList.bin file but not in raw data.'%(false_srcid,bindstid))
							flag = False
			
			# 再验证每一个halo的数据能不能在原始dgl数据中找到
			for p in range(self.partNum):
				halo[p] = torch.tensor(halo[p]).to('cuda:0')
				for ss in range(0,len(halo[p]),2):
					newsrcid = halo[p][ss].item()
					dstid = halo[p][ss+1].item()
					if (dstid not in partdict[p]) or (newsrcid not in partdict[p][dstid]):
						print('error!There is an edge(%d->%d) exists in halo%d.bin file but not in raw data.'%(newsrcid,dstid,p))
						flag = False

		if flag == True:
			print('0 errors in Graph in all parts')
		return flag

	def testFeat(self,datas):
		# datas: loadDGLData返回得到的各个分区图数据的列表
		print("start to test Feat....")
		flag = True
		for partIndex in range(self.partNum):
			# 加载原始分区数据
			subg,node_feat,node_type = datas[partIndex]
			# 取出特征tensor转换成numpy，注意原始数据格式里feat是二维数组，展开成一维比较
			featRawData = node_feat[node_type[0]+'/features'].detach().numpy().ravel()
			# 加载转换后的二进制数据到numpy，加载到的是一位
			featBinFile = self.newDataPath + 'part' + str(partIndex) + '/feat.bin'
			featBinData = np.fromfile(featBinFile, dtype=np.float32)
			# 两个float32的numpy数组做比较，一致说明转换正确
			comparison = featRawData == featBinData
			if comparison.all() == False:
				print('error in part %d feat'%partIndex)
				flag = False
		if flag == True:
			print('0 errors in feats in all parts.')
		return flag
	
	def main_test(self):
		flag = True
		datas = self.loadDGLData()
		if self.testLabel(datas) == False:
			flag = False
		if self.testID(datas) == False:
			flag = False
		if self.testFeat(datas) == False:
			flag = False
		if self.testGraph(datas) == False:
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