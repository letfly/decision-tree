#include <cstdio>
#include <cstdlib>
#include "pthread_pool.h"

static void * thread(void *arg) {
  PoolQueue *q;
  Pool *p = (Pool *) arg;

  while (!p->cancelled) {
    pthread_mutex_lock(&p->q_mtx);
    while (!p->cancelled && p->q == NULL) {
      pthread_cond_wait(&p->q_cnd, &p->q_mtx);
    }
    if (p->cancelled) {
      pthread_mutex_unlock(&p->q_mtx);
      return NULL;
    }
    q = p->q;
    p->q = q->next;
    p->end = (q == p->end ? NULL : p->end);
    pthread_mutex_unlock(&p->q_mtx);

    p->fn(q->arg);

    if (q->free) free(q->arg);
    free(q);
    q = NULL;

    pthread_mutex_lock(&p->q_mtx);
    p->remaining--;
    pthread_cond_broadcast(&p->q_cnd);
    pthread_mutex_unlock(&p->q_mtx);
  }

  return NULL;
}
Pool *pool_start(void *(*thread_func)(void *), unsigned int threads) {
  printf("pool_start..");
  Pool *p; // = (Pool *) malloc(sizeof(Pool)+(threads-1)*sizeof(pthread_t));
  int i;
  pthread_t p_threads[threads];

  pthread_mutex_init(&p->q_mtx, NULL);
  pthread_cond_init(&p->q_cnd, NULL);
  p->nthreads = threads;
  p->fn = thread_func;
  p->cancelled = 0;
  p->remaining = 0;
  p->end = NULL;
  p->q = NULL;

  for (i = 0; i < threads; ++i) pthread_create(&p_threads[i], NULL, thread, p);
  return p;
}

void pool_enquence(void *pool, void *arg, char free) {
  Pool *p = (Pool *) pool;
  PoolQueue *q = (PoolQueue *) malloc(sizeof(PoolQueue));
  q->arg = arg;
  q->next = NULL;
  q->free = free;

  pthread_mutex_lock(&p->q_mtx);
  if (p->end != NULL) p->end->next = q;
  if (p->q == NULL) p->q = q;
  p->end = q;
  ++p->remaining;
  pthread_cond_signal(&p->q_cnd);
  pthread_mutex_unlock(&p->q_mtx);
}

void pool_wait(void *pool) {
  Pool *p = (Pool *) pool;

  pthread_mutex_lock(&p->q_mtx);
  while (!p->cancelled && p->remaining) pthread_cond_wait(&p->q_cnd, &p->q_mtx);
  pthread_mutex_unlock(&p->q_mtx);
}

void pool_end(void *pool) {
  Pool *p = (Pool *)pool;
  PoolQueue *q;
  int i;

  p->cancelled = 1;

  pthread_mutex_lock(&p->q_mtx);
  pthread_cond_broadcast(&p->q_cnd);
  pthread_mutex_unlock(&p->q_mtx);

  for (i = 0; i < p->nthreads; ++i) pthread_join(p->threads[i], NULL);

  while (p->q != NULL) {
    q = p->q;
    p->q = q->next;

    if (q->free) free(q->arg);
    free(q);
  }
  free(p);
}
