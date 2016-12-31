gboost
=======
Creater: Yufan Fu: letflykid@gmail.com

General Purpose Gradient Boosting Library

Intention: A stand-alone efficient library to do machine learning in function

Planned key components (TODO):

(1) Gradient boosting models:
    - regression tree
    - linear model/lasso
(2) Objectives to support tasks:
    - regression
    - classification
    - ranking
    - matrix factorization
    - structured prediction
(3) OpenMP support for parallelization(optional)

File extension convention:
(1) .h are interface, utils and data structures, with detailed comment;
(2) .cc are implementations that will be compiled, with less comment;
(3) .hpp are implementations that will be included by .cc, with less comment
