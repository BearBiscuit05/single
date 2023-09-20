#!/bin/bash
RAW_DATA_DIR='/raid/bear/raw_dataset'
TW_RAW_DATA_DIR="${RAW_DATA_DIR}/uk-2007-05"
OUTPUT_DATA_DIR='/raid/bear/dataset/uk-2007-05'

download(){
  mkdir -p ${UK_RAW_DATA_DIR}
  if [ ! -e "${UK_RAW_DATA_DIR}/uk-2007-05.graph" ]; then
    pushd ${UK_RAW_DATA_DIR}
    wget http://data.law.di.unimi.it/webdata/uk-2007-05/uk-2007-05.graph
    wget http://data.law.di.unimi.it/webdata/uk-2007-05/uk-2007-05.properties
    popd
  elif [ ! -e "${UK_RAW_DATA_DIR}/uk-2007-05.properties" ]; then
    pushd ${UK_RAW_DATA_DIR}
    wget http://data.law.di.unimi.it/webdata/uk-2007-05/uk-2007-05.properties
    popd
  else
    echo "Binary file already downloaded."
  fi
}

generate_coo(){
  download
  if [ ! -e "${UK_RAW_DATA_DIR}/coo.bin" ]; then
    java -cp ./utils/mavenWeb/target/webgraph-0.1-SNAPSHOT.jar it.unimi.dsi.webgraph.BVGraph -o -O -L "${UK_RAW_DATA_DIR}/uk-2007-05"
    java -cp ./utils/mavenWeb/target/webgraph-0.1-SNAPSHOT.jar ddl.sgg.WebgraphDecoder "${UK_RAW_DATA_DIR}/uk-2007-05"
    mv ${UK_RAW_DATA_DIR}/uk-2007-05_coo.bin ${UK_RAW_DATA_DIR}/coo.bin
  else
    echo "COO already generated."
  fi
}


NUM_NODE=105896555
NUM_EDGE=3738733648
FEAT_DIM=768
NUM_CLASS=150
NUM_TRAIN_SET=1000000
NUM_VALID_SET=200000
NUM_TEST_SET=100000

mkdir -p $OUTPUT_DATA_DIR
dd if=/dev/zero of="${OUTPUT_DATA_DIR}/feats.bin" bs=4 count=$((NUM_NODE * FEAT_DIM))
dd if=/dev/zero of="${OUTPUT_DATA_DIR}/labels.bin" bs=8 count=$NUM_NODE