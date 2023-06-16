#!/bin/bash
set -e
# 运行命令 ./convert2txt.sh .

# npy2txt
if [ $# -lt 1 ]; then
    echo "请输入路径参数"
else
    input_path=$1
    datasetName="data"
    dataset_path="${input_path}/${datasetName}"
fi

if [ -d "${dataset_path}" ]; then
    echo "文件夹 $dataset_path 存在"
else
    echo "文件夹 $dataset_path 不存在"
    python partgraph.py
fi
savefile=processed
savepath="$1/${savefile}"
rm -rf ${savepath}
mkdir -p ${savepath}
python convert2coo.py ${datasetName} ogb-product ${savepath}


# txt2bin

g++ -o coo2csr coo2csrbin.cpp 
for ((i=0; i<4; i++))
do
    savebin="${savepath}/part${i}"
    mkdir "${savebin}"
    ./coo2csr "${savepath}/subg_${i}.txt" ${savebin}  
done

g++ -o coo2edge ./coo2edgebin.cpp
for ((i=0; i<4; i++))
do
    savebin="${savepath}/part${i}"
    mkdir -p "${savebin}"
    ./coo2edge "${savepath}/subg_${i}_bound_0.txt" "${savebin}/halo0.bin"  
    ./coo2edge "${savepath}/subg_${i}_bound_1.txt" "${savebin}/halo1.bin" 
    ./coo2edge "${savepath}/subg_${i}_bound_2.txt" "${savebin}/halo2.bin"  
    ./coo2edge "${savepath}/subg_${i}_bound_3.txt" "${savebin}/halo3.bin"  
done