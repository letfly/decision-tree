#!/bin/bash
g++ -Wall -O3 -msse2 main.cc io/io.cc -o main -I ./
# Map the data to features
#python examples/mapfeat.py
# Split train and test
#python examples/kfold.py examples/reg.txt 1
# Training and output the models
time ./main examples/reg.conf
