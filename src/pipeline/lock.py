import threading

sample_merge_con = threading.Condition()
merge_train_con = threading.Condition()
trainIDs = [i for i in range(10)]
sampleList = []
feat = []


def sample(sample_merge_con,trainIDs,sampleList):
    sample_merge_con.acquire()
    for info in trainIDs:
        sampleList.append(info*2)
    sample_merge_con.notify()
    sample_merge_con.release()

def merge(sample_merge_con,merge_train_con,sampleList,feat):
    sample_merge_con.acquire()
    merge_train_con.acquire()
    sample_merge_con.wait()
    for info in sampleList:
        feat.append([ info for i in range(10)])
    sample_merge_con.release()
    merge_train_con.notify()
    merge_train_con.release()

def train(merge_train_con,sampleList,feat):
    merge_train_con.acquire()
    merge_train_con.wait()
    print(sampleList)
    print(feat)
    merge_train_con.release()



 
