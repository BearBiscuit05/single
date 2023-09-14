#!/bin/bash

set -o errexit

layerList=(2 2 3 3)
fanoutList=("10,25" "10,15" "5,10,15" "10,10,10")
datasetNameList=("ogb-products" "Reddit" "ogb-papers100M")
MAXLOOP=10
MODELNAME="SAGE"
ExecuteDGLPath="/home/bear/workspace/singleGNN/src/train/dgl/dgl_train.py"
ExecutePYGPath="/home/bear/workspace/singleGNN/src/train/pyg/pyg_train.py"
output_file="baseline_log.txt"

length=${#fanoutList[@]}
for ((i = 0; i < length; i++)); do
    fanout="${fanoutList[i]}"
    layerNUM="${layerList[i]}"
    {
    echo "==============================================================" >> "$output_file"
    echo "${ExecuteDGLPath} --model ${MODELNAME} --fanout ${fanout} --layers ${layerNUM} \\
            --maxloop ${MAXLOOP} --dataset ogb-products" #>> "$output_file"
    echo "--------------------------------------------------------------" >> "$output_file"
    python ${ExecuteDGLPath} --model ${MODELNAME} --fanout ${fanout} --layers ${layerNUM} \
            --maxloop ${MAXLOOP} --dataset ogb-products #>> "$output_file"
    
    echo "==============================================================" >> "$output_file"
    echo "${ExecuteDGLPath} --model ${MODELNAME} --fanout ${fanout} --layers ${layerNUM} \
            --maxloop ${MAXLOOP} --dataset Reddit" #>> "$output_file"
    echo "--------------------------------------------------------------" >> "$output_file"
    python ${ExecuteDGLPath} --model ${MODELNAME} --fanout ${fanout} --layers ${layerNUM} \
            --maxloop ${MAXLOOP} --dataset Reddit #>> "$output_file"
    } >> "$output_file" &

    last_pid=$!

    echo "last pid : '${last_pid}'"
    # echo "==============================================================" >> "$output_file"
    # echo "${ExecuteDGLPath} --model ${MODELNAME} --fanout ${fanout} --layers ${layerNUM} \
    #         --maxloop ${MAXLOOP} --dataset ogb-papers100M" #>> "$output_file"
    # echo "--------------------------------------------------------------" >> "$output_file"
    # python ${ExecuteDGLPath} --model ${MODELNAME} --fanout ${fanout} --layers ${layerNUM} \
    #         --maxloop ${MAXLOOP} --dataset ogb-papers100M #>> "$output_file"
done

# fanoutList=("25,10" "15,10" "15,10,5" "10,10,10")
# for ((i = 0; i < length; i++)); do
#     fanout="${fanoutList[i]}"
#     layerNUM="${layerList[i]}"
#     echo "==============================================================" >> "$output_file"
#     echo "${ExecutePYGPath} --model ${MODELNAME} --fanout ${fanout} --layers ${layerNUM} \\
#             --maxloop ${MAXLOOP} --dataset ogb-products" #>> "$output_file"
#     echo "--------------------------------------------------------------" >> "$output_file"
#     python ${ExecutePYGPath} --model ${MODELNAME} --fanout ${fanout} --layers ${layerNUM} \
#             --maxloop ${MAXLOOP} --dataset ogb-products #>> "$output_file"
    
#     echo "==============================================================" >> "$output_file"
#     echo "${ExecutePYGPath} --model ${MODELNAME} --fanout ${fanout} --layers ${layerNUM} \
#             --maxloop ${MAXLOOP} --dataset Reddit" #>> "$output_file"
#     echo "--------------------------------------------------------------" >> "$output_file"
#     python ${ExecutePYGPath} --model ${MODELNAME} --fanout ${fanout} --layers ${layerNUM} \
#             --maxloop ${MAXLOOP} --dataset Reddit #>> "$output_file"
    
#     # echo "==============================================================" >> "$output_file"
#     # echo "${ExecutePYGPath} --model ${MODELNAME} --fanout ${fanout} --layers ${layerNUM} \
#     #         --maxloop ${MAXLOOP} --dataset ogb-papers100M" #>> "$output_file"
#     # echo "--------------------------------------------------------------" >> "$output_file"
#     # python ${ExecutePYGPath} --model ${MODELNAME} --fanout ${fanout} --layers ${layerNUM} \
#     #         --maxloop ${MAXLOOP} --dataset ogb-papers100M #>> "$output_file"
# done