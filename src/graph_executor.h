#ifndef OMLE_GRAPH_EXECUTOR_H_
#define OMLE_GRAPH_EXECUTOR_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "expression_eval.h"
#include "graph_node.h"
#include "model_base.h"
#include "runtime_tensor.h"
#include "thread_pool.h"

namespace omle::rt::impl {

struct OutputSpec {
  std::string name;
  int cols = 1;
};

class GraphExecutor final : public ModelBase {
 public:
  std::vector<std::string> input_names;
  std::vector<OutputSpec> output_specs;
  int total_output_cols = 0;
  std::vector<std::unique_ptr<GraphNode>> nodes;
  ConstantStore constants;
  UserFunctionMap user_functions;
  std::unique_ptr<ThreadPool> thread_pool;
  int min_parallel_rows = 64;

  int num_inputs() const override {
    return static_cast<int>(input_names.size());
  }
  int num_outputs() const override { return total_output_cols; }

  // Named path (single-threaded for now).
  std::unordered_map<std::string, omle::rt::Tensor> predict_named(
      const std::unordered_map<std::string, omle::rt::Tensor>& inputs,
      const std::vector<std::string>& output_filter = {}) const override;

  std::unordered_map<std::string, omle::rt::Tensor> predict_named_serial(
      const std::unordered_map<std::string, omle::rt::Tensor>& inputs,
      const std::vector<std::string>& output_filter = {}) const override;

  // Low-level flat float32 path (uses thread pool).
  void predict(const float* features, int n_samples,
               float* output) const override;
  void predict_serial(const float* features, int n_samples,
                      float* output) const override;

 private:
  void predict_chunk(const float* features, float* output, int row_start,
                     int row_end) const;

  void run_graph(
      const std::unordered_map<std::string, omle::rt::Tensor>& inputs,
      int row_start, int row_end,
      std::unordered_map<std::string, omle::rt::Tensor>& outputs,
      const std::vector<std::string>& output_filter = {}) const;
};

}  // namespace omle::rt::impl

#endif  // OMLE_GRAPH_EXECUTOR_H_
