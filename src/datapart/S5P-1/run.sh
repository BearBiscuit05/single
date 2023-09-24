#!/bin/bash
set -e

monitor_memory_usage() {
    echo "=============================================" >> $memory_usage_file
    local python_pid="$1"
    local memory_usage_file="$2"
    local interval="$3"

    local total_memory_usage_kb=0
    local peak_memory_usage_kb=0
    local count=0

    while ps -p $python_pid > /dev/null; do
        local memory_usage_kb=$(ps -o rss= -p $python_pid)
        local memory_usage_mb=$(echo "scale=2; $memory_usage_kb / 1024" | bc)
        local current_time=$(date +'%Y-%m-%d %H:%M:%S')
        # echo "$current_time, PID $python_pid, 内存占用: $memory_usage_mb MB" >> $memory_usage_file

        # 更新累计值和峰值
        total_memory_usage_kb=$((total_memory_usage_kb + memory_usage_kb))
        if [[ $memory_usage_kb -gt $peak_memory_usage_kb ]]; then
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

    echo "平均内存占用: $average_memory_usage_mb MB" >> $memory_usage_file
    echo "峰值内存占用: $((peak_memory_usage_kb / 1024)) MB" >> $memory_usage_file
}
interval=0.5
logMsg=""
memory_usage_file="memory_usage.txt"
/home/dzz/graphpartition/SGG/S5P_v3/S5P-1/build/bin/main &
lastPid=$!
logMsg="666" 
echo $logMsg >> $memory_usage_file
monitor_memory_usage $lastPid $memory_usage_file $interval