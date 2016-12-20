build:
	g++ -pthread -std=c++0x main.cc ../cart/matrix.cc ../cart/tree_node.cc ../cart/stats.cc forest.cc parallel_forest.cc pthread_pool.cc -o main -I ../cart/
	time ./main ../data/ex00.txt ../data/result.txt
