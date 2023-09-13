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
output_file="sgnn_log.txt"

# config
# fanoutList=("10,25" "5,10,15" "10,15" "10,10,10")
fanoutList=("10,25")
modelList=("SAGE" "GCN" "GAT")

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

# dgl
python ${ExecutePath} --file ${ConfigPath} --key_value \
        "framework='dgl'"

# for model in "${modelList[@]}"; do
#     for fanout in "${fanoutList[@]}"; do
#         for ((i = 0; i < length; i++)); do
#             datasetPath="${datasetPathList[i]}"
#             num="${PartNUM[i]}"
#             datasetName="${datasetNameList[i]}"
#             FeatLen="${FeatLenList[i]}"
#             Classes="${ClassesList[i]}"

#             echo "'${ExecutePath}' --file '${ConfigPath}' --key_value \\
#                 fanout=[${fanout}] model=${model} partNUM=${num} datasetpath=${datasetPath} \\
#                 dataset=${datasetName} featlen=${FeatLen} classes=${Classes}"
            
#             echo "==============================================================" >> "$output_file"
#             echo "'${ExecutePath}' --file '${ConfigPath}' --key_value \\
#                 fanout=[${fanout}] model=${model} partNUM=${num} datasetpath=${datasetPath} \\
#                 dataset=${datasetName} featlen=${FeatLen} classes=${Classes}" >> "$output_file"
#             echo "--------------------------------------------------------------" >> "$output_file"

#             python ${ExecutePath} --file ${ConfigPath} --key_value \
#                 "fanout=[${fanout}]" "model='${model}'" "partNUM=${num}" "dataset='${datasetName}'" \
#                 "featlen=${FeatLen}" "classes=${Classes}" "datasetpath='${datasetPath}'" >> "$output_file"
#             echo "--------------------------------------------------------------" >> "$output_file"
            
#             python ${SGNN_DGL_RUN} --json_path ${ConfigPath}

#         done
#     done
# done


# pyg
python ../config/modify.py --file ${ConfigPath} --key_value \
        "framework='pyg'"
fanoutList=("25,10" "15,10,5" "15,10" "10,10,10")
for model in "${modelList[@]}"; do
    for fanout in "${fanoutList[@]}"; do
        for ((i = 0; i < length; i++)); do
            datasetPath="${datasetPathList[i]}"
            num="${PartNUM[i]}"
            datasetName="${datasetNameList[i]}"
            FeatLen="${FeatLenList[i]}"
            Classes="${ClassesList[i]}"

            echo "'${ExecutePath}' --file '${ConfigPath}' --key_value \\
                fanout=[${fanout}] model=${model} partNUM=${num} datasetpath=${datasetPath} \\
                dataset=${datasetName} featlen=${FeatLen} classes=${Classes}"
            
            echo "==============================================================" >> "$output_file"
            echo "'${ExecutePath}' --file '${ConfigPath}' --key_value \\
                fanout=[${fanout}] model=${model} partNUM=${num} datasetpath=${datasetPath} \\
                dataset=${datasetName} featlen=${FeatLen} classes=${Classes}" >> "$output_file"
            echo "--------------------------------------------------------------" >> "$output_file"

            python ${ExecutePath} --file ${ConfigPath} --key_value \
                "fanout=[${fanout}]" "model='${model}'" "partNUM=${num}" "dataset='${datasetName}'" \
                "featlen=${FeatLen}" "classes=${Classes}" "datasetpath='${datasetPath}'" >> "$output_file"

            python ${SGNN_PYG_RUN} --json_path ${ConfigPath}
        done
    done
done