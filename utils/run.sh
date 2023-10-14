#!/bin/bash

java -cp ../utility/webgraph/target/webgraph-0.1-SNAPSHOT.jar it.unimi.dsi.webgraph.BVGraph -o -O -L "${UK_RAW_DATA_DIR}/uk-2006-05"
java -cp ../utility/webgraph/target/webgraph-0.1-SNAPSHOT.jar ipads.samgraph.webgraph.WebgraphDecoder "${UK_RAW_DATA_DIR}/uk-2006-05"