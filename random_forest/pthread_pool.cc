#include <cstdio>
#include <cstdlib> // free, malloc
#include <pthread.h>
#include "pthread_pool.h"

// A definition of a task in thread pool
struct pool_queue {
  void *arg;
  bool free;
  struct pool_queue *next;
};

/**
 * \brief The threadpool struct
 *
 * \param shutdown      Flag indicating if the pool is shutting down
 * \param fn            The function of task
 * \param remaining     Number of pending threads
 * \param nthreads      Number of threads
 * \param q             Array containing the task queue
 * \param end           The next task in the queues
 * \param q_mtx         A mutex for internal work
 * \param q_cnd         Condition variable to notify worker threads
 * \param threads       Array containing worker threads ID
 */
struct pool {
  bool shutdown;
  void *(*fn)(void *);
  unsigned int remaining;
  unsigned int nthreads;
  struct pool_queue *q;
  struct pool_queue *end;
  pthread_mutex_t q_mtx;
  pthread_cond_t q_cnd;
  pthread_t threads[1];
};

// Each thread in the thread pool runs in the function.
// The declaration static should only be used to make the function valid only
// within this file.
static void *thread(void *arg) {
  struct pool_queue *q;
  struct pool *p = (struct pool *) arg;

  while (!p->shutdown) {
    // Lock must be taken to wait on conditional variable
    pthread_mutex_lock(&p->q_mtx);

    // Use while in order to re-check the conditions in the wake
    while (!p->shutdown && p->q == NULL)
      // The service queue is empty, and the thread pool is not blocked when it is blocked here
      pthread_cond_wait(&p->q_cnd, &p->q_mtx);

    // Off processing
    if (p->shutdown) {
      pthread_mutex_unlock(&p->q_mtx);
      return NULL;
    }

    // Update q and end
    q = p->q;
    p->q = q->next;
    p->end = (q == p->end ? NULL : p->end);
    // Unlock
    pthread_mutex_unlock(&p->q_mtx);

    // Start running the task
    p->fn(q->arg);

    if (q->free) free(q->arg);
    free(q);
    q = NULL;

    pthread_mutex_lock(&p->q_mtx);
    // The thread will end and update the number of running threads
    --p->remaining;
    pthread_cond_broadcast(&p->q_cnd);
    pthread_mutex_unlock(&p->q_mtx);
  }

  return NULL;
}

void *pool_start(void *(*thread_func)(void *), unsigned int threads) {
  // Request memory to create a thread pool object and an array of threads
  struct pool *p = (struct pool*) malloc(sizeof(struct pool) + (threads-1) * sizeof(pthread_t));
  int i;

  // Initialize
  p->nthreads = threads;
  p->fn = thread_func;
  p->shutdown = 0;
  p->remaining = 0;
  p->q = NULL;
  p->end = NULL;
  // Initialize mutex and conditional variable first
  pthread_mutex_init(&p->q_mtx, NULL);
  pthread_cond_init(&p->q_cnd, NULL);

  // Creates the specified number of threads to run
  for (i = 0; i < threads; ++i) pthread_create(&p->threads[i], NULL, thread, p);

  return p;
}

void pool_enqueue(void *pool, void *arg, bool free) {
  struct pool *p = (struct pool *) pool;
  struct pool_queue *q = (struct pool_queue *) malloc(sizeof(struct pool_queue));
  q->arg = arg;
  q->next = NULL;
  q->free = free;

  // Mutex lock ownership must be acquired first
  pthread_mutex_lock(&p->q_mtx);

  // Calculate the location where the next task can be stored
  if (p->end != NULL) p->end->next = q;
  if (p->q == NULL) p->q = q;
  p->end = q;
  // Update remaining
  ++p->remaining;

  // A signal is issued indicating that a task has been added
  pthread_cond_signal(&p->q_cnd);

  // Release the mutex resource
  pthread_mutex_unlock(&p->q_mtx);
}

void pool_wait(void *pool) {
  struct pool *p = (struct pool *) pool;

  pthread_mutex_lock(&p->q_mtx);
  // The threads is remaining, and the thread pool is blocked when it is not closed here
  while (!p->shutdown && p->remaining)
    pthread_cond_wait(&p->q_cnd, &p->q_mtx);
  pthread_mutex_unlock(&p->q_mtx);
}

void pool_end(void *pool) {
  struct pool *p = (struct pool *) pool;
  struct pool_queue *q;
  int i;

  p->shutdown = 1;

  // Obtain a mutex resource
  pthread_mutex_lock(&p->q_mtx);
  // Wake up all worker threads
  pthread_cond_broadcast(&p->q_cnd);
  pthread_mutex_unlock(&p->q_mtx);

  // Join all worker thread
  for (i = 0; i < p->nthreads; ++i) pthread_join(p->threads[i], NULL);

  // Release thread,task queue,mutex,condition variable,the thread pool occupies the memory resource
  while (p->q != NULL) {
    q = p->q;
    p->q = q->next;

    if (q->free) free(q->arg);
    free(q);
  }

  free(p);
}
