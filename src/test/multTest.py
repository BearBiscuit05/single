import concurrent.futures

# 假设这是你的数据列表，初始为空
data = []

# 定义一个函数，模拟采样任务
def sample_task():
    # 在这里执行你的采样操作，并将采样结果添加到data列表中
    # 这里仅做示例，使用一个简单的字符串作为采样结果
    result = "sampled data"
    data.append(result)

# 定义一个函数，模拟释放数据并加载新数据的操作
def reload_data():
    # 在这里执行释放数据的操作
    data.clear()
    # 在这里执行加载新数据的操作
    # 这里仅做示例，使用一个简单的字符串列表作为新数据
    new_data = ["new data 1", "new data 2", "new data 3"]
    data.extend(new_data)

# 创建线程池，设置线程数为3
pool = concurrent.futures.ThreadPoolExecutor(max_workers=3)

# 设置采样次数和释放数据的阈值
sample_count = 10
reload_threshold = 5

for i in range(sample_count):
    # 提交采样任务给线程池
    future = pool.submit(sample_task)

    # 判断是否达到释放数据的阈值
    if (i + 1) % reload_threshold == 0:
        # 等待之前的采样任务完成
        concurrent.futures.wait([future])

        # 释放数据并加载新数据
        reload_data()

# 关闭线程池
pool.shutdown()
