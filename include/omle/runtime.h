#ifndef OMLE_RUNTIME_H_
#define OMLE_RUNTIME_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <variant>
#include <vector>

#include "status.h"
#include "tensor.h"

namespace omle::rt {

// -----------------------------------------------------------------------
// MeasureLevel — measurement level of a feature or target.
// -----------------------------------------------------------------------
enum class MeasureLevel : uint8_t {
  Unspecified = 0,
  Continuous = 1,
  Nominal = 2,
  Ordinal = 3,
  Flag = 4,
};

// -----------------------------------------------------------------------
// TaskType — high-level task type of a model.
// -----------------------------------------------------------------------
enum class TaskType : uint8_t {
  Unspecified = 0,
  Regression = 1,
  Binary = 2,
  Multiclass = 3,
  Clustering = 4,
  AnomalyDetection = 5,
};

// -----------------------------------------------------------------------
// OutputRole — high-level semantic role of a model output tensor.
// -----------------------------------------------------------------------
enum class OutputRole : uint8_t {
  Unspecified = 0,
  Prediction = 1,
  Probability = 2,
  Score = 3,
  Confidence = 4,
  StandardError = 5,
  StandardDev = 6,
  Residual = 7,
  TransformedValue = 8,
  EntityId = 9,
  Affinity = 10,
  Contribution = 11,
  Intermediate = 12,
};

// -----------------------------------------------------------------------
// InputSpec — metadata for one named model input.
// -----------------------------------------------------------------------
struct InputSpec {
  std::string name;
  DataType dtype = DataType::Unknown;
  std::vector<int64_t> shape;  // leading dim = batch (dynamic = -1)
};

// -----------------------------------------------------------------------
// OutputSpec — metadata for one named model output (InputSpec + role).
// -----------------------------------------------------------------------
struct OutputSpec : InputSpec {
  OutputRole role = OutputRole::Unspecified;
};

// -----------------------------------------------------------------------
// LoadOptions — configuration passed to Model::load().
// -----------------------------------------------------------------------
struct LoadOptions {
  int n_threads = 1;  // 0 = auto, 1 = single-threaded (default)
  int min_parallel_rows = 64;
  bool run_verification = true;  // run built-in verification cases after load
  bool run_warmup = true;        // execute warmup passes after load
};

// -----------------------------------------------------------------------
// Forward declaration
// -----------------------------------------------------------------------
class Session;

// -----------------------------------------------------------------------
// Model — immutable after construction, thread-safe.
// -----------------------------------------------------------------------
class Model {
 public:
  ~Model();
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;

  static StatusOr<std::unique_ptr<Model>> load(const std::string& path,
                                               const LoadOptions& opts = {});
  static StatusOr<std::unique_ptr<Model>> load(const void* data,
                                               std::size_t size,
                                               const LoadOptions& opts = {});

  const std::vector<InputSpec>& inputs() const;
  const std::vector<OutputSpec>& outputs() const;

  // Non-fatal advisories raised while loading this model — currently a schema
  // version this build does not recognise on a model carrying no verification
  // cases to prove otherwise. Empty for the ordinary case. Returned rather than
  // printed so an embedder can route them; the Python bindings turn each into a
  // warnings.warn().
  const std::vector<std::string>& warnings() const;
  int num_inputs() const;
  int num_outputs() const;

  std::unique_ptr<Session> create_session() const;

  // Named inputs → named outputs.  Thread-safe (uses thread pool if
  // configured). If `output_filter` is non-empty only those named outputs are
  // returned.
  StatusOr<std::unordered_map<std::string, Tensor>> predict(
      const std::unordered_map<std::string, Tensor>& inputs,
      const std::vector<std::string>& output_filter = {}) const;

  // Same as above but writes results into `outputs` in-place instead of
  // returning them.
  Status predict(const std::unordered_map<std::string, Tensor>& inputs,
                 std::unordered_map<std::string, Tensor>& outputs) const;

 private:
  Model();
  struct Impl;
  std::shared_ptr<Impl> impl_;
  friend class Session;
};

// -----------------------------------------------------------------------
// Session — per-thread execution context, not thread-safe.
// -----------------------------------------------------------------------
class Session {
 public:
  ~Session();
  Session(const Session&) = delete;
  Session& operator=(const Session&) = delete;

  // Bind a named input tensor for the next run().
  Session& bind_input(std::string name, Tensor t);

  // Bind a named output tensor to collect results into on the next run().
  // If no outputs are bound all graph outputs are collected.
  Session& bind_output(std::string name, Tensor t);

  // Remove all bound inputs / registered outputs.
  void clear_inputs();
  void clear_outputs();

  // Execute with the currently bound inputs.
  Status run();

  // Execute with an externally supplied input map instead of bound_inputs.
  // Inputs are accessed by const ref inside the graph — no tensor copies.
  // Used by the C API fast path to avoid per-call string-tensor allocation.
  Status run_with_inputs(const std::unordered_map<std::string, Tensor>& inputs);

  // Access results produced by the last run().
  StatusOr<Tensor> get_output(std::string_view name) const;
  const std::unordered_map<std::string, Tensor>& results() const;

  // Move all results out of the session, leaving it empty.
  // Called by the C API to take ownership without copying.
  std::unordered_map<std::string, Tensor> take_results();

  int num_inputs() const;
  int num_outputs() const;

 private:
  Session();
  struct Impl;
  std::unique_ptr<Impl> impl_;
  friend class Model;
};

}  // namespace omle::rt

#endif  // OMLE_RUNTIME_H_
