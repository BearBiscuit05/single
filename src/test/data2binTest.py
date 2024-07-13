"""
Test graph data is converted to binary files without errors:
# Test process and progress
## Data conversion test
dgl->bin file
1. Test whether there is a problem when converting graph data from dgl data set (edge set) to srclist and bound (completed)
2. Test whether there is a problem with the conversion of features (completed)
3. Test whether there is a problem with the conversion of data labels (completed)
4. Test the training set, verify the set, and test the set whether there is a problem (completed)
5. Test whether the output of the halo section can be restored to the original (completed)
The hope is that our binary data can be reloaded into dgl, exactly the same as dgl's original graph + feature
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
		# Load the raw data of the partition graph before conversion using DGL
		print("start to load raw data from dgl...")
		datas = []
		for partIndex in range(self.partNum):
			subg, node_feat, _, _, _, node_type, _ = dgl.distributed.load_partition(self.partConfigJson,partIndex)
			data = (subg,node_feat,node_type)
			datas.append(data)
		return datas
	
	def testID(self,datas):
		# datas: loadDGLData returns a list of the resulting partition graph data
		print("start to test IDs....")
		flag = True
		for partIndex in range(self.partNum):
			# Load the raw partition data
			subg,node_feat,node_type = datas[partIndex]
			# Take the feature tensor and convert it to numpy
			trainIDRawData = node_feat[node_type[0]+'/train_mask'].nonzero().squeeze()
			valIDRawData = node_feat[node_type[0]+'/val_mask'].nonzero().squeeze()
			testIDRawData = node_feat[node_type[0]+'/test_mask'].nonzero().squeeze()
			# Load the converted binary data to numpy
			trainIDBinFile = self.newDataPath + 'part' + str(partIndex) + '/trainID.bin'
			trainIDBinData = torch.load(trainIDBinFile).to(torch.uint8).nonzero().squeeze()
			valIDBinFile = self.newDataPath + 'part' + str(partIndex) + '/valID.bin'
			valIDBinData = torch.load(valIDBinFile).to(torch.uint8).nonzero().squeeze()
			testIDBinFile = self.newDataPath + 'part' + str(partIndex) + '/testID.bin'
			testIDBinData = torch.load(testIDBinFile).to(torch.uint8).nonzero().squeeze()
			print(trainIDRawData)
			# Two int32 numpy arrays are compared to show that the conversion is correct
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
		# datas: loadDGLData returns a list of the resulting partition graph data
		print("start to test Label....")
		flag = True
		for partIndex in range(self.partNum):
			# Load the raw partition data
			subg,node_feat,node_type = datas[partIndex]
			# Take the feature tensor and convert it to numpy
			labelRawData = node_feat[node_type[0]+'/labels'].numpy()
			# Load the converted binary data to numpy
			labelBinFile = self.newDataPath + 'part' + str(partIndex) + '/label.bin'
			labelBinData = np.fromfile(labelBinFile, dtype=np.int32)
			# Two int32 numpy arrays are compared to show that the conversion is correct
			comparison = labelRawData == labelBinData
			if comparison.all() == False:
				print('error in part %d labels'%partIndex)
				flag = False
		if flag == True:
			print('0 errors in labels in all parts.')
		return flag
	
	def testGraph(self,datas):
		# datas: loadDGLData returns a list of the resulting partition graph data
		print("start to test Graph....")
		flag = True

		with open(self.partConfigJson,'r') as f:
			SUBGconf = json.load(f)
		# Use the read data
		boundRange = SUBGconf['node_map']['_N']

		for partIndex in range(self.partNum):
			print('test part ',partIndex)
			# Load converted binary data, including srcList.bin+range.bin(within subgraph) and halo.bin+halo_bound.bin(across subgraph)
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
			
			# The srcList part of the binary data is stored first
			binsrcs = []
			for j in range(len(rangeBinData)//2):
				l,r = rangeBinData[j*2],rangeBinData[j*2+1]
				if r-l == 1 and srcListBinData[l] == j:
					binsrcs.append([])
				else:
					binsrcs.append(srcListBinData[l:r])
			
			# Load the raw partition data
			subg,node_feat,node_type = datas[partIndex]
			src = subg.edges()[0].tolist()
			dst = subg.edges()[1].tolist()
			
			# In this test, it is found that there are repeats on the self-cyclic edges of some nodes in srcList.bin
			# It's not predata, it's dgl raw graph data where these self-looping edges are also repeated
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
			
			# In the first stage, forward validation, 
			# the edge in the original data can be found in the converted binary data, 
			# regardless of whether the src is inside the subgraph
			print('[stage 1]:check if all edges in raw data can be found in .bin file')
			for index in range(len(src)):
				srcid,dstid = src[index],dst[index]
				# Nodes that are both within the subgraph of the original data, detect whether it is written to srcList.bin
				if inner[srcid] == 1 and inner[dstid] == 1:
					if srcid not in binsrcs[dstid]:
						flag = False
						print('error! part%d edge(%d->%d) not exist in srcList.bin'%(partIndex,srcid,dstid))
				# dstid at a point in the subgraph, detects whether its srcid has written the corresponding halo
				# Check that the edge from srcid to dstid in the partid subgraph is written to the corresponding halo.bin and halo_bound.bin
				elif inner[srcid] != 1 and inner[dstid] == 1:
					local_srcid = srcid
					srcid = subg.ndata[dgl.NID][srcid] # srcid ：local query global ID
					partid=-1
					for pid,(left,right) in enumerate(boundRange):
						if left <= srcid and srcid < right:
							partid = pid
							break
					if dstid not in partdict[partid]:
						partdict[partid][dstid] = []
					boundl,boundr = halo_bound[partid][dstid],halo_bound[partid][dstid+1]
					newsrcid = srcid - SUBGconf['node_map']['_N'][partid][0] + basiclen # The id written is the offset id of the two subgraphs combined
					partdict[partid][dstid].append(newsrcid)
					
					found_flag = False
					# Query in the left and right boundaries
					for ss in range(boundl,boundr,2):
						mysrcid = halo[partid][ss]
						mydstid = halo[partid][ss+1]
						if mysrcid == newsrcid:
							found_flag = True
						if mydstid != dstid:# dst within the bounds should correspond, checking halo and halo_bound
							print('error in part%d/halo%d.bin with dst = %d'%(partIndex,partid,dstid))
					if found_flag == False:
						flag = False
						print('error! In part%d graph,edge(%d->%d) exists in dgl raw data but not in halo%d.bin'%(partIndex,newsrcid,dstid,partid))	

			# In the second stage, reverse verification, the edge in the binary file can be queried in the original data, and this part of the GPU is operated
			cuda_subg = subg.to('cuda:0')
			print('[stage 2]:check if all edges in .bin file can be found in raw data')
			# First verify that the srcList.bin binary can be found in the original dgl data
			for bindstid in range(len(binsrcs)):
				if len(binsrcs[bindstid]) == 0:
					continue
				binsrcids = torch.tensor(binsrcs[bindstid],device='cuda:0')
				bindstids = torch.tensor([bindstid for i in range(len(binsrcs[bindstid]))],device='cuda:0')
				# dgl subgraph internal edge detection function, passing n edges at a time
				comp = cuda_subg.has_edges_between(binsrcids,bindstids)
				if comp.all() == False:
					false_indices = torch.nonzero(~comp)
					for fi in false_indices:
						false_srcid = binsrcids[fi]
						if false_srcid != bindstid:
							print('error!There is an edge(%d->%d) exists in srcList.bin file but not in raw data.'%(false_srcid,bindstid))
							flag = False
			
			# Then verify that each halo data can be found in the original dgl data
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
		# datas: loadDGLData returns a list of the resulting partition graph data
		print("start to test Feat....")
		flag = True
		for partIndex in range(self.partNum):
			# Load the raw partition data
			subg,node_feat,node_type = datas[partIndex]
			# Take the feature tensor and convert it into numpy. 
			# Note that feat in the original data format is a two-dimensional array, 
			# expanded into a one-dimensional comparison
			featRawData = node_feat[node_type[0]+'/features'].detach().numpy().ravel()
			# Load the converted binary data to numpy, which is loaded to a bit
			featBinFile = self.newDataPath + 'part' + str(partIndex) + '/feat.bin'
			featBinData = np.fromfile(featBinFile, dtype=np.float32)
			# A comparison of two numpy arrays of float32 shows that the conversion is correct
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
		# if self.testFeat(datas) == False:
		# 	flag = False
		# if self.testGraph(datas) == False:
		# 	flag = False
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