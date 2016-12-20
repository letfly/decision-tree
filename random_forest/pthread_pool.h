#ifndef PTHREAD_POOL_H
#define PTHREAD_POOL_H
void *pool_start(void *(*thread_func)(void *), unsigned int threads);
void pool_enquence(void *pool, void *arg, char free);
void pool_wait(void *pool);
#endif
