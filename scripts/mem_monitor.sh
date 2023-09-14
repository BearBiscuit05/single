#!/bin/bash

# 启动Python程序并将其输出重定向到文件
python your_script.py > output.txt &

# 获取Python程序的PID
python_pid=$!

# 存储内存占用数据的文件名
memory_usage_file="memory_usage.txt"

# 设置监测时间间隔（秒）
interval=10

# 监测内存占用
while ps -p $python_pid > /dev/null; do
    # 使用ps命令获取进程的内存占用（以KB为单位），并使用awk将其转换为MB
    memory_usage_kb=$(ps -o rss= -p $python_pid)
    memory_usage_mb=$(echo "scale=2; $memory_usage_kb / 1024" | bc)

    # 获取当前时间
    current_time=$(date +'%Y-%m-%d %H:%M:%S')

    # 写入内存占用数据到文件
    echo "$current_time, PID $python_pid, 内存占用: $memory_usage_mb MB" >> $memory_usage_file

    # 休眠一段时间
    sleep $interval
done

echo "Python程序已终止，停止监测内存占用."
