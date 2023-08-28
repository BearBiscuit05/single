import os
import random

def find_nextId(folder_path,partNUM):
    idMap={}
    for i in range(partNUM):
        folder_path = "../../data/papers100M_64/part"+str(i)
        idMap[i] = []
        for filename in os.listdir(folder_path):
            if filename.startswith("halo") and filename.endswith(".bin"):
                try:
                    x = int(filename[len("halo"):-len(".bin")])
                    idMap[i].append(x)
                except:
                    continue
    return idMap

def custom_sort(partNUM, idMap,epoch):
    sorted_numbers = []
    lastid = 0
    for loop in range(epoch):
        used_numbers = set()
        tmp = []
        for idx in range(0,partNUM):
            if idx == 0:
                num = lastid
            else:
                num = tmp[-1]
            candidates = idMap[num]
            available_candidates = [int(candidate) for candidate in candidates if int(candidate) not in used_numbers]                
            if available_candidates:
                chosen_num = random.choice(available_candidates)
                tmp.append(chosen_num)
                used_numbers.add(chosen_num)
            else:
                for i in range(partNUM):
                    if i not in used_numbers:
                        available_candidates.append(i)
                chosen_num = random.choice(available_candidates)
                tmp.append(chosen_num)
                used_numbers.add(chosen_num)
        sorted_numbers.append(tmp)
        lastid = tmp[-1]
    print(sorted_numbers)
    return sorted_numbers

idMap = find_nextId(32)
custom_sort(32, idMap)