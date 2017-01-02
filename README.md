## Classification and Regression Trees
### Running the program
```
cd cart
make
```

## Random Forest
### Running the program
```
cd random_forest
make
```

### Required Parameters
- -t ../data/train_file
    - input train file to matrix
- -s ../data/result_file
    - output predict labels to result file

### Optional Parameters
- -p 16
    - use 16 threads in a thread pool to train trees
- -n 1000
    - use 1000 trees in the forest
- -f 30
    - use a subset of 30 features for each tree

## Gbrt
### Running the program
```
cd gbrt
make
```
### Required Parameters
### Optional Parameters
