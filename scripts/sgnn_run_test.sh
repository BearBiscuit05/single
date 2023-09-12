#!/bin/bash
set -o errexit

# 文件路径
ConfigPath="/home/bear/workspace/singleGNN/config/test_config.json"
ProductsPath="/home/bear/workspace/singleGNN/data/products"
RedditPath="/home/bear/workspace/singleGNN/data/reddit"
PaperPath="/home/bear/workspace/singleGNN/data/papers100M"
ExecutePath="/home/bear/workspace/singleGNN/config/modify.py"

output_file="sgnn_log.txt"

# config
fanoutList=("10,25" "5,10,15" "10,15" "10,10,10")
modelList=("SAGE" "GCN" "GAT")

datasetPathList=("$ProductsPath" "$RedditPath" "$PaperPath")
datasetNameList=("products_4" "reddit_8" "papers100M_64")
PartNUM=(4 8 32)
FeatLenList=(100 602 128)
ClassesList=(47 41 172)
length=${#datasetPathList[@]}

# dgl
for model in "${modelList[@]}"; do
    for fanout in "${fanoutList[@]}"; do
        for ((i = 0; i < length; i++)); do
            datasetPath="${datasetPathList[i]}"
            num="${PartNUM[i]}"
            datasetName="${datasetNameList[i]}"
            FeatLen="${FeatLenList[i]}"
            Classes="${ClassesList[i]}"

            echo "Executing: '${ExecutePath}' --file '${ConfigPath}' --key_value \\
                fanout=[${fanout}] model=${model} partNUM=${num} datasetpath=${datasetPath} \\
                dataset=${datasetName} featlen=${FeatLen} classes=${Classes}"
            
            echo "==============================================================" >> "$output_file"
            echo "Executing: '${ExecutePath}' --file '${ConfigPath}' --key_value \\
                fanout=[${fanout}] model=${model} partNUM=${num} datasetpath=${datasetPath} \\
                dataset=${datasetName} featlen=${FeatLen} classes=${Classes}"
            echo "--------------------------------------------------------------" >> "$output_file"

            python ${ExecutePath} --file ${ConfigPath} --key_value \
                "fanout=[${fanout}]" "model='${model}'" "partNUM=${num}" "dataset='${datasetName}'" \
                "featlen=${FeatLen}" "classes=${Classes}" "datasetpath='${datasetPath}'" >> "$output_file"

        done
    done
done


# pyg
python ../config/modify.py --file ${ConfigPath} --key_value \
        "framework='pyg'"

for model in "${modelList[@]}"; do
    for fanout in "${fanoutList[@]}"; do
        for ((i = 0; i < length; i++)); do
            datasetPath="${datasetPathList[i]}"
            num="${PartNUM[i]}"
            datasetName="${datasetNameList[i]}"
            FeatLen="${FeatLenList[i]}"
            Classes="${ClassesList[i]}"

            echo "Executing: '${ExecutePath}' --file '${ConfigPath}' --key_value \\
                fanout=[${fanout}] model=${model} partNUM=${num} datasetpath=${datasetPath} \\
                dataset=${datasetName} featlen=${FeatLen} classes=${Classes}"
            
            echo "==============================================================" >> "$output_file"
            echo "Executing: '${ExecutePath}' --file '${ConfigPath}' --key_value \\
                fanout=[${fanout}] model=${model} partNUM=${num} datasetpath=${datasetPath} \\
                dataset=${datasetName} featlen=${FeatLen} classes=${Classes}"
            echo "--------------------------------------------------------------" >> "$output_file"

            python ${ExecutePath} --file ${ConfigPath} --key_value \
                "fanout=[${fanout}]" "model='${model}'" "partNUM=${num}" "dataset='${datasetName}'" \
                "featlen=${FeatLen}" "classes=${Classes}" "datasetpath='${datasetPath}'" >> "$output_file"

        done
    done
done