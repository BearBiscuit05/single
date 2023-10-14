#!/bin/bash

java -cp ./mavenWeb/target/webGraphTest-1.0-SNAPSHOT.jar it.unimi.dsi.webgraph.BVGraph -o -O -L "/raid/bear/wget_paper/uk2007/uk-2007-05"
java -cp ./mavenWeb/target/webGraphTest-1.0-SNAPSHOT.jar ddl.sgg.WebgraphDecoder "/raid/bear/wget_paper/uk2007/uk-2007-05"