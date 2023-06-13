import json

data = {
    "name": "John",
    "age": 30,
    "city": "New York"
}

# 将数据写入JSON文件
with open("data.json", "w") as file:
    json.dump(data, file)

import json

# 从JSON文件中读取数据
with open("data.json", "r") as file:
    data = json.load(file)

# 打印读取到的数据
print(data)
