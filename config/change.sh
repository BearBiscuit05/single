#!/bin/bash
python configChange.py --pattern "./dgl_*.json" --key fanout --value [10,25]
