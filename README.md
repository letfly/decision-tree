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

## Gradient Boosting Regression Tree
### Running the program
```
cd gradient_boosting
./run.sh
```
### Required Parameters
- train_path = "examples/train_reg.txt"
- eval[test] = "examples/test_reg.txt"
- test_reg = "examples/test_reg.txt"

### Optional Parameters
- booster = gbtree
- objective = reg:linear

- eta = 1.0
- gamma = 1.0
- min_child_weight = 1
- max_depth = 3

- num_round = 2
- save_period = 0
