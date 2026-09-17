#include "graph_executor.h"

#include <algorithm>
#include <cstring>
#include <vector>

#include "omle/status.h"

namespace omle::rt::impl {

// ----------------------------------------------------------------
// Low-level float32 chunk path (unchanged from original).
// ----------------------------------------------------------------
void GraphExecutor::predict_chunk(const float* features, float* output,
                                  int row_start, int row_end) const {
  const int nf = static_cast<int>(input_names.size());
  const int n_rows = row_end - row_start;

  UserFunctionContext ufc(user_functions);
  ValueStore vs(constants);

  for (int f = 0; f < nf; ++f) {
    Tensor col(n_rows, 1);
    for (int r = 0; r < n_rows; ++r)
      col.f32_ptr()[r] = features[(row_start + r) * nf + f];
    vs.put(input_names[f], std::move(col));
  }

  for (const auto& node : nodes)
    if (auto st = node->execute(vs, n_rows); !st.ok())
      throw std::runtime_error(st.message());

  const int no = static_cast<int>(output_specs.size());
  int col_off = 0;
  for (int j = 0; j < no; ++j) {
    const OutputSpec& spec = output_specs[j];
    if (!vs.has(spec.name))
      throw std::runtime_error("graph: output '" + spec.name +
                               "' not produced");
    const Tensor& t = vs.get(spec.name);
    const int stride = t.n_cols;
    for (int r = 0; r < n_rows; ++r)
      std::memcpy(output + (row_start + r) * total_output_cols + col_off,
                  t.row(r), stride * sizeof(float));
    col_off += stride;
  }
}

void GraphExecutor::predict_serial(const float* features, int n_samples,
                                   float* output) const {
  if (n_samples > 0) predict_chunk(features, output, 0, n_samples);
}

void GraphExecutor::predict(const float* features, int n_samples,
                            float* output) const {
  if (n_samples <= 0) return;
  const int n_threads = thread_pool ? thread_pool->size() : 1;
  const int threshold = n_threads * min_parallel_rows;
  if (n_threads > 1 && n_samples >= threshold) {
    const int chunk = (n_samples + n_threads - 1) / n_threads;
    thread_pool->parallel_for(n_threads, [&](int t) {
      const int start = t * chunk;
      if (start >= n_samples) return;
      predict_chunk(features, output, start,
                    std::min(start + chunk, n_samples));
    });
  } else {
    predict_chunk(features, output, 0, n_samples);
  }
}

// ----------------------------------------------------------------
// Named path implementation.
// ----------------------------------------------------------------
void GraphExecutor::run_graph(
    const std::unordered_map<std::string, omle::rt::Tensor>& inputs,
    int row_start, int row_end,
    std::unordered_map<std::string, omle::rt::Tensor>& result,
    const std::vector<std::string>& output_filter) const {
  const int n_rows = row_end - row_start;
  UserFunctionContext ufc(user_functions);
  ValueStore vs(constants);

  // Check whether inputs already supply each slot by name.
  bool slots_present = input_names.empty() || inputs.count(input_names[0]) > 0;

  if (!slots_present && inputs.size() == 1) {
    // Single flat tensor supplied — split columns into individual named slots.
    const auto& src = inputs.begin()->second;
    const omle::rt::Tensor flat =
        (src.dtype != omle::rt::DataType::Float32 || src.is_sparse())
            ? src.to_float32()
            : src;
    const int nf = static_cast<int>(input_names.size());
    for (int f = 0; f < nf; ++f) {
      Tensor col(n_rows, 1);
      for (int r = 0; r < n_rows; ++r)
        col.f32_ptr()[r] = flat.row(row_start + r)[f];
      vs.put(input_names[f], std::move(col));
    }
  } else {
    for (const auto& name : input_names) {
      auto it = inputs.find(name);
      if (it == inputs.end())
        throw std::runtime_error("predict: missing input '" + name + "'");

      const auto& src = it->second;

      // String tensors are passed through as-is; numeric tensors are
      // converted to Float32 dense so the compute graph can use them.
      if (src.is_string()) {
        if (row_start == 0 && row_end == src.n_rows) {
          vs.put(name, src);
        } else {
          // Slice the string tensor row-by-row.
          int nc = src.n_cols;
          omle::rt::Tensor slice = omle::rt::Tensor::strings(n_rows, nc);
          for (int r = 0; r < n_rows; ++r)
            for (int c = 0; c < nc; ++c)
              slice.str_at(r, c) = src.str_at(row_start + r, c);
          vs.put(name, std::move(slice));
        }
      } else {
        const omle::rt::Tensor t =
            (src.dtype != omle::rt::DataType::Float32 || src.is_sparse())
                ? src.to_float32()
                : src;

        // Always produce an owned tensor so nodes can safely modify in-place.
        // For view tensors (from_buffer) t.row() via const is valid; the slice
        // path materializes even when row_start==0 covers the whole tensor.
        if (row_start == 0 && row_end == t.n_rows && !t.is_view()) {
          vs.put(name, t);
        } else {
          Tensor slice(n_rows, t.n_cols);
          std::memcpy(
              slice.f32_ptr(), t.row(row_start),
              static_cast<std::size_t>(n_rows) * t.n_cols * sizeof(float));
          vs.put(name, std::move(slice));
        }
      }
    }
  }

  for (const auto& node : nodes)
    if (auto st = node->execute(vs, n_rows); !st.ok())
      throw std::runtime_error(st.message());

  for (const auto& spec : output_specs) {
    if (!output_filter.empty() &&
        std::find(output_filter.begin(), output_filter.end(), spec.name) ==
            output_filter.end())
      continue;
    if (!vs.has(spec.name))
      throw std::runtime_error("predict: output '" + spec.name +
                               "' not produced");
    result[spec.name] = vs.get(spec.name);
  }
}

// Concatenate dense numeric tensor chunks vertically (row-wise).
static omle::rt::Tensor vcat_tensors(std::vector<omle::rt::Tensor> parts) {
  int total_rows = 0;
  for (const auto& p : parts) total_rows += p.n_rows;
  const int n_cols = parts[0].n_cols;
  const omle::rt::DataType dt = parts[0].dtype;
  omle::rt::Tensor out = omle::rt::Tensor::dense(dt, total_rows, n_cols);
  int off = 0;
  for (const auto& p : parts) {
    const std::size_t n = static_cast<std::size_t>(p.n_rows) * n_cols;
    if (dt == omle::rt::DataType::Float64)
      std::memcpy(out.f64_ptr() + static_cast<std::size_t>(off) * n_cols,
                  p.f64_ptr(), n * sizeof(double));
    else
      std::memcpy(out.f32_ptr() + static_cast<std::size_t>(off) * n_cols,
                  p.f32_ptr(), n * sizeof(float));
    off += p.n_rows;
  }
  return out;
}

std::unordered_map<std::string, omle::rt::Tensor> GraphExecutor::predict_named(
    const std::unordered_map<std::string, omle::rt::Tensor>& inputs,
    const std::vector<std::string>& output_filter) const {
  if (inputs.empty()) return {};
  const int n_rows = inputs.begin()->second.n_rows;
  if (n_rows <= 0) return {};

  const int n_threads = thread_pool ? thread_pool->size() : 1;
  const int threshold = n_threads * min_parallel_rows;

  if (n_threads <= 1 || n_rows < threshold) {
    std::unordered_map<std::string, omle::rt::Tensor> result;
    run_graph(inputs, 0, n_rows, result, output_filter);
    return result;
  }

  // Parallel: split rows evenly across threads, each with its own ValueStore.
  const int chunk = (n_rows + n_threads - 1) / n_threads;
  std::vector<std::unordered_map<std::string, omle::rt::Tensor>> parts(
      n_threads);

  thread_pool->parallel_for(n_threads, [&](int t) {
    const int start = t * chunk;
    if (start >= n_rows) return;
    run_graph(inputs, start, std::min(start + chunk, n_rows), parts[t],
              output_filter);
  });

  // Filter non-empty parts (trailing threads beyond n_rows get nothing).
  std::vector<int> valid;
  valid.reserve(n_threads);
  for (int t = 0; t < n_threads; ++t)
    if (!parts[t].empty()) valid.push_back(t);

  if (valid.empty()) return {};
  if (valid.size() == 1) return std::move(parts[valid[0]]);

  // Concatenate per-output-name tensor chunks.
  std::unordered_map<std::string, omle::rt::Tensor> result;
  result.reserve(parts[valid[0]].size());
  for (const auto& [name, _] : parts[valid[0]]) {
    std::vector<omle::rt::Tensor> name_parts;
    name_parts.reserve(valid.size());
    for (int t : valid) name_parts.push_back(std::move(parts[t][name]));
    result[name] = vcat_tensors(std::move(name_parts));
  }
  return result;
}

std::unordered_map<std::string, omle::rt::Tensor>
GraphExecutor::predict_named_serial(
    const std::unordered_map<std::string, omle::rt::Tensor>& inputs,
    const std::vector<std::string>& output_filter) const {
  return predict_named(inputs, output_filter);
}

}  // namespace omle::rt::impl
