# Add function rules

## setup.py

In normal cases, you do not need to modify the file. If additional files are added, modify the file

## signn.h

Function declarations required for cuda internal builds

## sample_node.cpp

Used to interact with torch and bound to pybind11

Both sample_node.cpp and signn.h must be changed at the same time

## xx.cu

The implementation of cuda internal function is constructed