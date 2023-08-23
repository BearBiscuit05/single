#!/bin/bash
cd ../src/train/dgl
python gcn.py --mode mixed --fanout [10,25] --layers 2 --dataset ogb-products
python gcn.py --mode mixed --fanout [5,10,15] --layers 3 --dataset ogb-products
python gcn.py --mode mixed --fanout [10,15] --layers 2 --dataset ogb-products
python gcn.py --mode mixed --fanout [10,10,10] --layers 3 --dataset ogb-products
python gcn.py --mode mixed --fanout [10,25] --layers 2 --dataset Reddit
python gcn.py --mode mixed --fanout [5,10,15] --layers 3 --dataset Reddit
python gcn.py --mode mixed --fanout [10,15] --layers 2 --dataset Reddit
python gcn.py --mode mixed --fanout [10,10,10] --layers 3 --dataset Reddit

cd ../pyg
python gcn.py --fanout [25,10] --layers 2 --dataset ogb-products
python gcn.py --fanout [15,10,5] --layers 3 --dataset ogb-products
python gcn.py --fanout [15,10] --layers 2 --dataset ogb-products
python gcn.py --fanout [10,10,10] --layers 3 --dataset ogb-products
python gcn.py --fanout [25,10] --layers 2 --dataset Reddit
python gcn.py --fanout [15,10,5] --layers 3 --dataset Reddit
python gcn.py --fanout [15,10] --layers 2 --dataset Reddit
python gcn.py --fanout [10,10,10] --layers 3 --dataset Reddit

cd ../sgnn
python dgl_gcn.py --mode mixed --fanout [10,25] --layers 2 --dataset ogb-products --json_path ../../../config/dgl_products_gcn.json
python dgl_gcn.py --mode mixed --fanout [5,10,15] --layers 3 --dataset ogb-products --json_path ../../../config/dgl_products_gcn.json
python dgl_gcn.py --mode mixed --fanout [10,15] --layers 2 --dataset ogb-products --json_path ../../../config/dgl_products_gcn.json
python dgl_gcn.py --mode mixed --fanout [10,10,10] --layers 3 --dataset ogb-products --json_path ../../../config/dgl_products_gcn.json
python dgl_gcn.py --mode mixed --fanout [10,25] --layers 2 --dataset Reddit --json_path ../../../config/dgl_reddit_8.json
python dgl_gcn.py --mode mixed --fanout [5,10,15] --layers 3 --dataset Reddit --json_path ../../../config/dgl_reddit_8.json
python dgl_gcn.py --mode mixed --fanout [10,15] --layers 2 --dataset Reddit --json_path ../../../config/dgl_reddit_8.json
python dgl_gcn.py --mode mixed --fanout [10,10,10] --layers 3 --dataset Reddit --json_path ../../../config/dgl_reddit_8.json

python pyg_gcn.py --mode mixed --fanout [25,10] --layers 2 --dataset ogb-products --json_path ../../../config/pyg_products_gcn.json
python pyg_gcn.py --mode mixed --fanout [15,10,5] --layers 3 --dataset ogb-products --json_path ../../../config/pyg_products_gcn.json
python pyg_gcn.py --mode mixed --fanout [15,10] --layers 2 --dataset ogb-products --json_path ../../../config/pyg_products_gcn.json
python pyg_gcn.py --mode mixed --fanout [10,10,10] --layers 3 --dataset ogb-products --json_path ../../../config/pyg_products_gcn.json
python pyg_gcn.py --mode mixed --fanout [25,10] --layers 2 --dataset Reddit --json_path ../../../config/pyg_reddit_8.json
python pyg_gcn.py --mode mixed --fanout [15,10,5] --layers 3 --dataset Reddit --json_path ../../../config/pyg_reddit_8.json
python pyg_gcn.py --mode mixed --fanout [15,10] --layers 2 --dataset Reddit --json_path ../../../config/pyg_reddit_8.json
python pyg_gcn.py --mode mixed --fanout [10,10,10] --layers 3 --dataset Reddit --json_path ../../../config/pyg_reddit_8.json