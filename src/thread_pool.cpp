#include "thread_pool.h"

#include <algorithm>

namespace omle::rt::impl {

ThreadPool::ThreadPool(int n_threads) {
  workers_.reserve(n_threads);
  for (int i = 0; i < n_threads; ++i)
    workers_.emplace_back([this] { worker_loop(); });
}

ThreadPool::~ThreadPool() {
  {
    std::lock_guard<std::mutex> lk(mu_);
    stop_ = true;
    ++generation_;
  }
  cv_.notify_all();
  for (auto& w : workers_) w.join();
}

void ThreadPool::worker_loop() {
  int local_gen = generation_ - 1;  // behind on entry

  while (true) {
    std::unique_lock<std::mutex> lk(mu_);
    // Sleep until a new generation arrives or shutdown is requested.
    cv_.wait(lk, [&] { return stop_ || generation_ != local_gen; });
    if (stop_) return;
    local_gen = generation_;

    // Claim and execute tasks until this generation is exhausted.
    while (next_task_ < n_tasks_) {
      int task = next_task_++;
      ++active_;
      lk.unlock();

      (*fn_)(task);

      lk.lock();
      if (--active_ == 0 && next_task_ >= n_tasks_)
        cv_.notify_all();  // wake the main thread (and any idle workers)
    }
  }
}

void ThreadPool::parallel_for(int n_tasks, const std::function<void(int)>& fn) {
  if (n_tasks <= 0) return;
  {
    std::lock_guard<std::mutex> lk(mu_);
    fn_ = &fn;
    n_tasks_ = n_tasks;
    next_task_ = 0;
    active_ = 0;
    ++generation_;
  }
  cv_.notify_all();

  std::unique_lock<std::mutex> lk(mu_);
  cv_.wait(lk, [&] { return active_ == 0 && next_task_ >= n_tasks_; });
}

}  // namespace omle::rt::impl
