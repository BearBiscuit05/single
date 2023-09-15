#!/bin/bash
set -o errexit

# 文件路径
ConfigPath="/home/bear/workspace/singleGNN/config/train_config.json"
ProductsPath="/home/bear/workspace/singleGNN/data/products"
RedditPath="/home/bear/workspace/singleGNN/data/reddit"
PaperPath="/home/bear/workspace/singleGNN/data/papers100M"
ExecutePath="/home/bear/workspace/singleGNN/config/modify.py"

SGNN_DGL_RUN="/home/bear/workspace/singleGNN/src/train/sgnn/sgnn_dgl_train.py"
SGNN_PYG_RUN="/home/bear/workspace/singleGNN/src/train/sgnn/sgnn_pyg_train.py"

# config
# fanoutList=("10,25" "5,10,15" "10,15" "10,10,10")
fanoutList=("10,25")
#modelList=("SAGE" "GCN" "GAT")
modelList=("SAGE")
# datasetPathList=("$ProductsPath" "$RedditPath" "$PaperPath")
# datasetNameList=("products_4" "reddit_8" "papers100M_64")
# PartNUM=(4 8 32)
# FeatLenList=(100 602 128)
# ClassesList=(47 41 172)

datasetPathList=("$ProductsPath" "$RedditPath")
datasetNameList=("products_4" "reddit_8")
PartNUM=(4 8)
FeatLenList=(100 602)
ClassesList=(47 41)
length=${#datasetPathList[@]}

memory_usage_file="memory_usage.txt"
interval=2
logMsg=""
monitor_memory_usage() {
    local python_pid="$1"
    echo "$logMsg"
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
        echo "$current_time, PID $python_pid , 内存占用: $memory_usage_mb MB" >> $memory_usage_file

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



# dgl
python ${ExecutePath} --file ${ConfigPath} --key_value \
        "framework='dgl'"

for model in "${modelList[@]}"; do
    for fanout in "${fanoutList[@]}"; do
        if [[ "$model" == "GAT" && ("$fanout" == "10,10,10" || "$fanout" == "5,10,15") ]]; then
            continue 
        fi

        for ((i = 0; i < length; i++)); do
            datasetPath="${datasetPathList[i]}"
            num="${PartNUM[i]}"
            datasetName="${datasetNameList[i]}"
            FeatLen="${FeatLenList[i]}"
            Classes="${ClassesList[i]}"

            echo "'${ExecutePath}' --file '${ConfigPath}' --key_value \\
                fanout=[${fanout}] model=${model} partNUM=${num} datasetpath=${datasetPath} \\
                dataset=${datasetName} featlen=${FeatLen} classes=${Classes}"
            
            echo "==============================================================" 
            echo "'${ExecutePath}' --file '${ConfigPath}' --key_value \\
                fanout=[${fanout}] model=${model} partNUM=${num} datasetpath=${datasetPath} \\
                dataset=${datasetName} featlen=${FeatLen} classes=${Classes}" 
            echo "--------------------------------------------------------------" 

            python ${ExecutePath} --file ${ConfigPath} --key_value \
                "fanout=[${fanout}]" "model='${model}'" "partNUM=${num}" "dataset='${datasetName}'" \
                "featlen=${FeatLen}" "classes=${Classes}" "datasetpath='${datasetPath}'" 
            echo "--------------------------------------------------------------" 
            
            python ${SGNN_DGL_RUN} --json_path ${ConfigPath} &

            last_pid=$!
            echo "last pid : '${last_pid}'"
            logMsg="'${ExecutePath}' --file '${ConfigPath}' --key_value \\
                fanout=[${fanout}] model=${model} partNUM=${num} datasetpath=${datasetPath} \\
                dataset=${datasetName} featlen=${FeatLen} classes=${Classes}"
            monitor_memory_usage $last_pid
        done
    done
done


# pyg
python ../config/modify.py --file ${ConfigPath} --key_value \
        "framework='pyg'"
#fanoutList=("25,10" "15,10,5" "15,10" "10,10,10")
fanoutList=("25,10")
for model in "${modelList[@]}"; do
    for fanout in "${fanoutList[@]}"; do
        if [[ "$model" == "GAT" && ("$fanout" == "10,10,10" || "$fanout" == "15,10,5") ]]; then
            continue 
        fi

        for ((i = 0; i < length; i++)); do
            datasetPath="${datasetPathList[i]}"
            num="${PartNUM[i]}"
            datasetName="${datasetNameList[i]}"
            FeatLen="${FeatLenList[i]}"
            Classes="${ClassesList[i]}"

            echo "'${ExecutePath}' --file '${ConfigPath}' --key_value \\
                fanout=[${fanout}] model=${model} partNUM=${num} datasetpath=${datasetPath} \\
                dataset=${datasetName} featlen=${FeatLen} classes=${Classes}"
            
            echo "==============================================================" 
            echo "'${ExecutePath}' --file '${ConfigPath}' --key_value \\
                fanout=[${fanout}] model=${model} partNUM=${num} datasetpath=${datasetPath} \\
                dataset=${datasetName} featlen=${FeatLen} classes=${Classes}" 
            echo "--------------------------------------------------------------" 

            python ${ExecutePath} --file ${ConfigPath} --key_value \
                "fanout=[${fanout}]" "model='${model}'" "partNUM=${num}" "dataset='${datasetName}'" \
                "featlen=${FeatLen}" "classes=${Classes}" "datasetpath='${datasetPath}'" 

            python ${SGNN_PYG_RUN} --json_path ${ConfigPath} &

            last_pid=$!
            echo "last pid : '${last_pid}'"
            logMsg="'${ExecutePath}' --file '${ConfigPath}' --key_value \\
                fanout=[${fanout}] model=${model} partNUM=${num} datasetpath=${datasetPath} \\
                dataset=${datasetName} featlen=${FeatLen} classes=${Classes}"
            monitor_memory_usage $last_pid
        done
    done
done