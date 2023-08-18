#！/bin/bash
rm -rf build
rm -rf dist
rm -rf signn.egg-info
# python3 setup.py install > build.txt 2>&1
python3 setup.py install > build.txt