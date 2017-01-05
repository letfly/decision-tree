#!/bin/bash
# Map the data to features
python mapfeat.py
# Split train and test
python kfold.py reg.txt 1
# Training and output the models
time ../main reg.conf
