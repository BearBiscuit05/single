#!/bin/bash
#productsPath="/home/bear/workspace/singleGNN/data/products"
productsPath="/home/bear/workspace/singleGNN/data/products"
python modify.py --file "test_config.json" --key_value "fanout=[22,21]" "partNUM=8" "framework='${productsPath}'"
