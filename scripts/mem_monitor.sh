#!/bin/bash

interval=10
monitor_memory_usage() {
    local python_pid="$1"
    local memory_usage_file="$2"
    # local interval="$3"
    local logMsg="$3"

    local total_memory_usage_kb=0
    local peak_memory_usage_kb=0
    local count=0

    while ps -p $python_pid > /dev/null; do
        local memory_usage_kb=$(ps -o rss= -p $python_pid)
        local memory_usage_mb=$(echo "scale=2; $memory_usage_kb / 1024" | bc)
        local current_time=$(date +'%Y-%m-%d %H:%M:%S')
        echo "$current_time, PID $python_pid, 内存占用: $memory_usage_mb MB" >> $memory_usage_file

        # 更新累计值和峰值
        total_memory_usage_kb=$((total_memory_usage_kb + memory_usage_kb))
        if [ $memory_usage_kb -gt $peak_memory_usage_kb ]; then
            peak_memory_usage_kb=$memory_usage_kb
        fi
        count=$((count + 1))
        sleep $interval
    done

    if [ $count -gt 0 ]; then
        local average_memory_usage_kb=$((total_memory_usage_kb / count))
        local average_memory_usage_mb=$(echo "scale=2; $average_memory_usage_kb / 1024" | bc)
    else
        local average_memory_usage_mb=0
    fi

    echo "平均内存占用: $average_memory_usage_mb MB"
    echo "峰值内存占用: $((peak_memory_usage_kb / 1024)) MB"
}


python your_script.py > output.txt &
python_pid=$!
memory_usage_file="memory_usage.txt"
monitor_memory_usage $python_pid $memory_usage_file $interval
echo "Python程序已终止,停止监测内存占用."
