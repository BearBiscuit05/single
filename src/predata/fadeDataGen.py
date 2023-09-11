import random

# 创建4个空数组
arrays = [[] for _ in range(4)]

# 随机生成0-99的100个数并打乱顺序
numbers = list(range(100))
random.shuffle(numbers)

# 将打乱后的数字分到4个数组中
for i, number in enumerate(numbers):
    array_index = i % 4
    arrays[array_index].append(number)

# 随机生成一些边（数组内部）
for i in range(4):
    for j in range(5):
        src = random.choice(arrays[i])
        dest = random.choice(arrays[i])
        print(f"Edge within Array {i}: {src} -> {dest}")

# 随机生成一些边（数组内部和非数组内部）
for i in range(4):
    for j in range(5):
        src = random.choice(arrays[i])
        dest_array = (i + 1) % 4  # 选择不同的数组
        dest = random.choice(arrays[dest_array])
        print(f"Edge between Array {i} and Array {dest_array}: {src} -> {dest}")
