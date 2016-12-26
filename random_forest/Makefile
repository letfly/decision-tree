build:
	g++ -pthread -std=c++0x main.cc ../cart/matrix.cc ../cart/tree_node.cc ../cart/stats.cc forest.cc parallel_forest.cc pthread_pool.cc -o main -I ../cart/
	time ./main -t ../data/ex00.txt -p 4 -n 2 -f 2
