import json
import numpy as np

"""
graph.bin       int32
feat.bin        float32
labels.bin      int64
trainIds.bin    int64
# optional
valIds.bin      int64
testIds.bin     int64
"""


JSONPATH = "/home/bear/workspace/single-gnn/datasetInfo.json"
with open(JSONPATH, 'r') as file:
    data = json.load(file)


for NAME in data:
    PATH = data[NAME]['rawFilePath']
    edgeNUM = data[NAME]['edges']
    nodeNUM = data[NAME]['nodes']
    trainNUM = data[NAME]['trainNUM']
    featLen = data[NAME]['featLen']

    graph = np.fromfile(PATH+'/graph.bin',dtype=np.int32)
    feat = np.fromfile(PATH+'/feat.bin',dtype=np.float32)
    label = np.fromfile(PATH+'/labels.bin',dtype=np.int64)
    trainIds = np.fromfile(PATH+'/trainIds.bin',dtype=np.int64)

    assert graph.reshape(-1,2).shape[0] == edgeNUM , "edge file error ,please check 'graph.bin' file with int32"
    assert feat.reshape(-1,featLen).shape[0] == nodeNUM , "feat file error ,please check 'feat.bin' file with float32"
    assert label.shape[0] == nodeNUM , "label file error ,please check 'labels.bin' file with int64"
    assert trainIds.shape[0] == trainNUM , "trainIds file error ,please check 'trainIds.bin' file with int64"
    print(f"{NAME} dataset files checked over with no error!")