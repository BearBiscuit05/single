import psutil
import time
import sys


if len(sys.argv) != 2:
    print("Usage: python monitor_memory.py <PID>")
    exit(1)
try:
    process_pid = int(sys.argv[1])
except ValueError:
    print("Invalid PID. Please enter a valid PID.")
    exit(1)


def get_process_by_pid(pid):
    try:
        process = psutil.Process(pid)
        return process
    except psutil.NoSuchProcess:
        print(f"Process with PID {pid} does not exist.")
        return None

# 监测进程内存占用并写入文件
def monitor_memory_usage_and_write_to_file(process, interval_seconds, output_file):
    if not process:
        return

    with open(output_file, 'w') as f:
        while True:
            try:
                # 获取进程内存占用信息
                memory_info = process.memory_info()
                rss = memory_info.rss  # 常驻内存
                vms = memory_info.vms  # 虚拟内存

                # 写入内存占用信息到文件
                f.write(f"RSS: {rss} bytes, VMS: {vms} bytes\n")
                f.flush()  # 确保写入立即生效

                # 等待一定时间间隔
                time.sleep(interval_seconds)
            except psutil.NoSuchProcess:
                # 进程不存在，退出循环
                print("Process has terminated.")
                break
            except KeyboardInterrupt:
                # 如果用户中断程序，退出循环
                break

if __name__ == "__main__":
    process = get_process_by_pid(process_pid)

    if process:
        # 设置监测时间间隔（秒）
        interval_seconds = 3  # 替换成你需要的时间间隔
        # 设置输出文件路径
        output_file = "memory_usage.txt"  # 替换成你希望的文件路径
        monitor_memory_usage_and_write_to_file(process, interval_seconds, output_file)
