build:
	g++ -pthread -std=c++0x main.cc ../cart/matrix.cc ../cart/tree_node.cc ../cart/stats.cc forest.cc parallel_forest.cc pthread_pool.cc -o main -I ../
	time ./main -t ../data/09_train.csv -s ../data/test.csv -r ../data/result.csv -p 1 -n 2 -f 2
