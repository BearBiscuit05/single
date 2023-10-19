#!/bin/bash

set -o errexit

# layerList=(2 2 3 3)
# fanoutList=("10,25" "10,15" "5,10,15" "10,10,10")
# datasetNameList=("ogb-products" "Reddit" "ogb-papers100M")
# datasetNameList=("ogb-products" "Reddit")
MAXLOOP=10
modelList=("SAGE" "GCN" "GAT")
# MODELNAME="SAGE"
ExecuteDGLPath="/home/bear/workspace/singleGNN/src/train/dgl/dgl_train.py"
ExecutePYGPath="/home/bear/workspace/singleGNN/src/train/pyg/pyg_train.py"


memory_usage_file="memory_usage.txt"
interval=3
logMsg=""
monitor_memory_usage() {
    local python_pid="$1"
    echo "=============================================" >> $memory_usage_file
    echo "=============================================" >> $memory_usage_file
    echo "=============================================" >> $memory_usage_file
    echo "$logMsg"  >> $memory_usage_file

    local total_memory_usage_kb=0
    local peak_memory_usage_kb=0
    local count=0

    while ps -p $python_pid > /dev/null; do
        local memory_usage_kb=$(ps -o rss= -p $python_pid)
        local memory_usage_mb=$(echo "scale=2; $memory_usage_kb / 1024" | bc)
        local current_time=$(date +'%Y-%m-%d %H:%M:%S')
        # echo "$current_time, PID $python_pid , 内存占用: $memory_usage_mb MB" >> $memory_usage_file

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

    
    echo "平均内存占用: $average_memory_usage_mb MB" >> $memory_usage_file
    echo "峰值内存占用: $((peak_memory_usage_kb / 1024)) MB" >> $memory_usage_file
}

layerList=(3 3 2)
fanoutList=("5,10,15" "10,10,10" "10,25")

# for MODELNAME in "${modelList[@]}"; do
#     length=${#fanoutList[@]}
#     for ((i = 0; i < length; i++)); do
#         fanout="${fanoutList[i]}"
#         layerNUM="${layerList[i]}"
#         if [[ "$MODELNAME" == "GAT" && ("$fanout" == "10,10,10" || "$fanout" == "5,10,15") ]]; then
#             continue 
#         fi

#         echo "==============================================================" 
#         echo "${ExecuteDGLPath} --model ${MODELNAME} --fanout ${fanout} --layers ${layerNUM} \\
#                 --maxloop ${MAXLOOP} --dataset uk-2006-05" 
#         echo "--------------------------------------------------------------" 
#         python ${ExecuteDGLPath} --model ${MODELNAME} --fanout ${fanout} --layers ${layerNUM} \
#                 --maxloop ${MAXLOOP} --dataset uk-2006-05 & 
        
#         last_pid=$!
#         echo "last pid : '${last_pid}'"
#         logMsg="${ExecuteDGLPath} --model ${MODELNAME} --fanout ${fanout} --layers ${layerNUM} \\
#                 --maxloop ${MAXLOOP} --dataset uk-2006-05"
#         monitor_memory_usage $last_pid

#         # echo "==============================================================" 
#         # echo "${ExecuteDGLPath} --model ${MODELNAME} --fanout ${fanout} --layers ${layerNUM} \\
#         #         --maxloop ${MAXLOOP} --dataset twitter" 
#         # echo "--------------------------------------------------------------" 
#         # python ${ExecuteDGLPath} --model ${MODELNAME} --fanout ${fanout} --layers ${layerNUM} \
#         #         --maxloop ${MAXLOOP} --dataset twitter & 

#         # last_pid=$!
#         # echo "last pid : '${last_pid}'"
#         # logMsg="${ExecuteDGLPath} --model ${MODELNAME} --fanout ${fanout} --layers ${layerNUM} \\
#         #         --maxloop ${MAXLOOP} --dataset twitter"
#         # monitor_memory_usage $last_pid


        
#         # echo "==============================================================" 
#         # echo "${ExecuteDGLPath} --model ${MODELNAME} --fanout ${fanout} --layers ${layerNUM} \\
#         #         --maxloop ${MAXLOOP} --dataset ogb-papers100M" 
#         # echo "--------------------------------------------------------------" 
#         # python ${ExecuteDGLPath} --model ${MODELNAME} --fanout ${fanout} --layers ${layerNUM} \
#         #         --maxloop ${MAXLOOP} --dataset ogb-papers100M & 

#         # last_pid=$!
#         # echo "last pid : '${last_pid}'"
#         # logMsg="${ExecuteDGLPath} --model ${MODELNAME} --fanout ${fanout} --layers ${layerNUM} \\
#         #         --maxloop ${MAXLOOP} --dataset ogb-papers100M"
#         # monitor_memory_usage $last_pid

#     done
# done


fanoutList=("15,10,5" "10,10,10" "25,10")

for MODELNAME in "${modelList[@]}"; do
    length=${#fanoutList[@]}
    for ((i = 0; i < length; i++)); do
        fanout="${fanoutList[i]}"
        layerNUM="${layerList[i]}"

        echo "==============================================================" 
        echo "${ExecutePYGPath} --model ${MODELNAME} --fanout ${fanout} --layers ${layerNUM} \\
                --maxloop ${MAXLOOP} --dataset uk-2006-05" 
        echo "--------------------------------------------------------------" 
        python ${ExecutePYGPath} --model ${MODELNAME} --fanout ${fanout} --layers ${layerNUM} \
                --maxloop ${MAXLOOP} --dataset uk-2006-05 & 
        
        last_pid=$!
        echo "last pid : '${last_pid}'"
        logMsg="${ExecutePYGPath} --model ${MODELNAME} --fanout ${fanout} --layers ${layerNUM} \\
                --maxloop ${MAXLOOP} --dataset uk-2006-05"
        monitor_memory_usage $last_pid $memory_usage_file $logMsg

        # echo "==============================================================" 
        # echo "${ExecutePYGPath} --model ${MODELNAME} --fanout ${fanout} --layers ${layerNUM} \\
        #         --maxloop ${MAXLOOP} --dataset twitter" 
        # echo "--------------------------------------------------------------" 
        # python ${ExecutePYGPath} --model ${MODELNAME} --fanout ${fanout} --layers ${layerNUM} \
        #         --maxloop ${MAXLOOP} --dataset twitter & 

        # last_pid=$!
        # echo "last pid : '${last_pid}'"
        # logMsg="${ExecutePYGPath} --model ${MODELNAME} --fanout ${fanout} --layers ${layerNUM} \\
        #         --maxloop ${MAXLOOP} --dataset twitter"
        # monitor_memory_usage $last_pid $logMsg
        
        # echo "==============================================================" 
        # echo "${ExecutePYGPath} --model ${MODELNAME} --fanout ${fanout} --layers ${layerNUM} \
        #         --maxloop ${MAXLOOP} --dataset ogb-papers100M" 
        # echo "--------------------------------------------------------------" 
        # python ${ExecutePYGPath} --model ${MODELNAME} --fanout ${fanout} --layers ${layerNUM} \
        #         --maxloop ${MAXLOOP} --dataset ogb-papers100M 
    done
done