#ifndef PTHREAD_POOL_H
#define PTHREAD_POOL_H
#include <pthread.h>

void pool_enquence(void *pool, void *arg, char free);
void pool_wait(void *pool);
void pool_end(void *pool);
struct PoolQueue {
  void *arg;
  char free;
  struct PoolQueue *next;
};

struct Pool {
  char cancelled;
  void *(*fn)(void *);
  unsigned int remaining;
  unsigned int nthreads;
  PoolQueue *q;
  PoolQueue *end;
  pthread_mutex_t q_mtx;
  pthread_cond_t q_cnd;
  pthread_t *threads;
};
Pool *pool_start(void *(*thread_func)(void *), unsigned int threads);
#endif
