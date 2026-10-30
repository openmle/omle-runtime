#ifndef OMLE_THREAD_POOL_H_
#define OMLE_THREAD_POOL_H_

#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace omle::rt::impl {

// Fixed-size thread pool for parallel_for patterns.
// Threads are spawned once at construction and joined at destruction.
//
// Dispatch protocol uses a generation counter so workers never
// confuse tasks from successive parallel_for calls.
class ThreadPool {
 public:
  explicit ThreadPool(int n_threads);
  ~ThreadPool();

  int size() const { return static_cast<int>(workers_.size()); }

  // Execute fn(task_id) for task_id in [0, n_tasks), distributing tasks
  // across worker threads.  Blocks until every task has finished.
  void parallel_for(int n_tasks, const std::function<void(int)>& fn);

 private:
  void worker_loop();

  std::vector<std::thread> workers_;
  std::mutex mu_;
  std::condition_variable cv_;  // shared by dispatch and done waits

  const std::function<void(int)>* fn_ = nullptr;
  int n_tasks_ = 0;
  int next_task_ = 0;   // next task index to claim
  int active_ = 0;      // tasks grabbed but not yet finished
  int generation_ = 0;  // incremented on every parallel_for call
  bool stop_ = false;
};

}  // namespace omle::rt::impl

#endif  // OMLE_THREAD_POOL_H_
