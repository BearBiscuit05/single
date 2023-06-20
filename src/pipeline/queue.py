import threading
from queue import Queue

# 创建两个管道
pipe1 = Queue()
pipe2 = Queue()

def generate_array():
    array = [1, 2, 3, 4, 5]
    pipe1.put(array)  # 将生成的数组放入 pipe1

def calculate_array():
    array = pipe1.get()  # 从 pipe1 获取数组
    calculated_array = [x * 2 for x in array]  # 对数组进行计算
    pipe2.put(calculated_array)  # 将计算后的数组放入 pipe2

def print_array():
    calculated_array = pipe2.get()  # 从 pipe2 获取数组
    print("Printed Array:", calculated_array)  # 打印数组

if __name__ == "__main__":
    t1 = threading.Thread(target=generate_array)
    t2 = threading.Thread(target=calculate_array)
    t3 = threading.Thread(target=print_array)

    # 启动线程，按顺序启动
    t1.start()
    t1.join()  # 等待 t1 线程结束
    t2.start()
    t2.join()  # 等待 t2 线程结束
    t3.start()
    t3.join()  # 等待 t3 线程结束
