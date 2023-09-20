#!/bin/bash
RAW_DATA_DIR='/raid/bear/raw_dataset'
TW_RAW_DATA_DIR="${RAW_DATA_DIR}/twitter"
OUTPUT_DATA_DIR='/raid/bear/dataset/twitter'
# 下载源数据

download(){
  mkdir -p ${TW_RAW_DATA_DIR}
  if [ ! -e "${TW_RAW_DATA_DIR}/twitter-2010.graph" ]; then
    pushd ${TW_RAW_DATA_DIR}
    wget http://data.law.di.unimi.it/webdata/twitter-2010/twitter-2010.graph
    wget http://data.law.di.unimi.it/webdata/twitter-2010/twitter-2010.properties
    popd
  elif [ ! -e "${TW_RAW_DATA_DIR}/twitter-2010.properties" ]; then
    pushd ${TW_RAW_DATA_DIR}
    wget http://data.law.di.unimi.it/webdata/twitter-2010/twitter-2010.properties
    popd
  else
    echo "Binary file already downloaded."
  fi
}

generate_coo(){
  download
  if [ ! -e "${TW_RAW_DATA_DIR}/coo.bin" ]; then
    java -cp ./utils/mavenWeb/target/webgraph-0.1-SNAPSHOT.jar it.unimi.dsi.webgraph.BVGraph -o -O -L "${TW_RAW_DATA_DIR}/twitter-2010"
    java -cp ./utils/mavenWeb/target/webgraph-0.1-SNAPSHOT.jar ddl.sgg.WebgraphDecoder "${TW_RAW_DATA_DIR}/twitter-2010"
    mv ${TW_RAW_DATA_DIR}/twitter-2010_coo.bin ${TW_RAW_DATA_DIR}/coo.bin
  else
    echo "COO already generated."
  fi
}


NUM_NODE=41652230
NUM_EDGE=1468365182
FEAT_DIM=768
NUM_CLASS=150
NUM_TRAIN_SET=416500
NUM_VALID_SET=100000
NUM_TEST_SET=200000

mkdir -p $OUTPUT_DATA_DIR
# 生成特征数据集文件
dd if=/dev/zero of="${OUTPUT_DATA_DIR}/feats.bin" bs=4 count=$((NUM_NODE * FEAT_DIM))
dd if=/dev/zero of="${OUTPUT_DATA_DIR}/labels.bin" bs=8 count=$NUM_NODE
# 生成训练集/验证集/测试
