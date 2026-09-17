#include "omle/runtime.h"

#include <numeric>
#include <stdexcept>

#include "model_base.h"
#include "model_loader.h"

namespace omle::rt {

struct Model::Impl {
  std::shared_ptr<impl::ModelBase> executor;
  std::vector<InputSpec> inputs;
  std::vector<OutputSpec> outputs;
  int cached_num_inputs{0};
  int cached_num_outputs{0};
};

struct Session::Impl {
  std::shared_ptr<const Model::Impl> model;
  std::unordered_map<std::string, Tensor> bound_inputs;
  std::unordered_map<std::string, Tensor>
      bound_outputs;  // keyset doubles as output filter
  std::unordered_map<std::string, Tensor> last_results;
};

// -----------------------------------------------------------------------
// Internal helpers
// -----------------------------------------------------------------------

static int compute_num_inputs(const std::shared_ptr<impl::ModelBase>& executor,
                              const std::vector<InputSpec>& specs) noexcept {
  if (!specs.empty()) {
    int total = 0;
    for (const auto& s : specs) {
      int slots = 1;
      for (std::size_t d = 1; d < s.shape.size(); ++d)
        slots *= static_cast<int>(s.shape[d]);
      total += slots;
    }
    if (total > 0) return total;
  }
  return executor->num_inputs();
}

static int compute_num_outputs(const std::shared_ptr<impl::ModelBase>& executor,
                               const std::vector<OutputSpec>& specs) noexcept {
  if (!specs.empty()) {
    int total = 0;
    for (const auto& s : specs) {
      int slots = 1;
      for (std::size_t d = 1; d < s.shape.size(); ++d)
        slots *= static_cast<int>(s.shape[d]);
      total += slots;
    }
    if (total > 0) return total;
  }
  return executor->num_outputs();
}

static Status status_from_current_exception() noexcept {
  try {
    throw;
  } catch (const std::exception& e) {
    return {ErrorCode::InvalidGraph, e.what()};
  } catch (...) {
    return {ErrorCode::InvalidGraph, "unknown error"};
  }
}

// -----------------------------------------------------------------------
// Model
// -----------------------------------------------------------------------
Model::Model() : impl_(std::make_shared<Impl>()) {}
Model::~Model() = default;

StatusOr<std::unique_ptr<Model>> Model::load(const std::string& path,
                                             const LoadOptions& opts) {
  try {
    auto m = std::unique_ptr<Model>(new Model());
    auto loaded = impl::load_from_file(path, opts);
    m->impl_->executor =
        std::shared_ptr<impl::ModelBase>(std::move(loaded.executor));
    m->impl_->inputs = std::move(loaded.inputs);
    m->impl_->outputs = std::move(loaded.outputs);
    m->impl_->cached_num_inputs =
        compute_num_inputs(m->impl_->executor, m->impl_->inputs);
    m->impl_->cached_num_outputs =
        compute_num_outputs(m->impl_->executor, m->impl_->outputs);
    return m;
  } catch (...) {
    return status_from_current_exception();
  }
}

StatusOr<std::unique_ptr<Model>> Model::load(const void* data, std::size_t size,
                                             const LoadOptions& opts) {
  try {
    auto m = std::unique_ptr<Model>(new Model());
    auto loaded = impl::load_from_bytes(data, size, opts);
    m->impl_->executor =
        std::shared_ptr<impl::ModelBase>(std::move(loaded.executor));
    m->impl_->inputs = std::move(loaded.inputs);
    m->impl_->outputs = std::move(loaded.outputs);
    m->impl_->cached_num_inputs =
        compute_num_inputs(m->impl_->executor, m->impl_->inputs);
    m->impl_->cached_num_outputs =
        compute_num_outputs(m->impl_->executor, m->impl_->outputs);
    return m;
  } catch (...) {
    return status_from_current_exception();
  }
}

const std::vector<InputSpec>& Model::inputs() const { return impl_->inputs; }
const std::vector<OutputSpec>& Model::outputs() const { return impl_->outputs; }

int Model::num_inputs() const { return impl_->cached_num_inputs; }
int Model::num_outputs() const { return impl_->cached_num_outputs; }

std::unique_ptr<Session> Model::create_session() const {
  auto s = std::unique_ptr<Session>(new Session());
  s->impl_->model = impl_;
  return s;
}

StatusOr<std::unordered_map<std::string, Tensor>> Model::predict(
    const std::unordered_map<std::string, Tensor>& inputs,
    const std::vector<std::string>& output_filter) const {
  try {
    return impl_->executor->predict_named(inputs, output_filter);
  } catch (...) {
    return status_from_current_exception();
  }
}

Status Model::predict(const std::unordered_map<std::string, Tensor>& inputs,
                      std::unordered_map<std::string, Tensor>& outputs) const {
  try {
    std::vector<std::string> keys;
    keys.reserve(outputs.size());
    for (const auto& [k, _] : outputs) keys.push_back(k);
    auto result = impl_->executor->predict_named(inputs, keys);
    for (auto& [k, v] : result)
      outputs.insert_or_assign(std::move(k), std::move(v));
    return {};
  } catch (...) {
    return status_from_current_exception();
  }
}

// -----------------------------------------------------------------------
// Session
// -----------------------------------------------------------------------
Session::Session() : impl_(std::make_unique<Impl>()) {}
Session::~Session() = default;

Session& Session::bind_input(std::string name, Tensor t) {
  impl_->bound_inputs.insert_or_assign(std::move(name), std::move(t));
  return *this;
}

Session& Session::bind_output(std::string name, Tensor t) {
  impl_->bound_outputs.insert_or_assign(std::move(name), std::move(t));
  return *this;
}

void Session::clear_inputs() { impl_->bound_inputs.clear(); }

void Session::clear_outputs() { impl_->bound_outputs.clear(); }

Status Session::run() {
  try {
    std::vector<std::string> filter;
    if (!impl_->bound_outputs.empty()) {
      filter.reserve(impl_->bound_outputs.size());
      for (const auto& [k, _] : impl_->bound_outputs) filter.push_back(k);
    }
    auto fresh = impl_->model->executor->predict_named_serial(
        impl_->bound_inputs, filter);
    impl_->last_results.clear();
    for (auto& [k, v] : fresh)
      impl_->last_results.insert_or_assign(std::move(k), std::move(v));
    return {};
  } catch (...) {
    return status_from_current_exception();
  }
}

Status Session::run_with_inputs(
    const std::unordered_map<std::string, Tensor>& inputs) {
  try {
    std::vector<std::string> filter;
    if (!impl_->bound_outputs.empty()) {
      filter.reserve(impl_->bound_outputs.size());
      for (const auto& [k, _] : impl_->bound_outputs) filter.push_back(k);
    }
    auto fresh = impl_->model->executor->predict_named_serial(inputs, filter);
    impl_->last_results.clear();
    for (auto& [k, v] : fresh)
      impl_->last_results.insert_or_assign(std::move(k), std::move(v));
    return {};
  } catch (...) {
    return status_from_current_exception();
  }
}

StatusOr<Tensor> Session::get_output(std::string_view name) const {
  for (const auto& [k, v] : impl_->last_results)
    if (k == name) return v;
  return {ErrorCode::OutputNotFound, "omle: output '" + std::string(name) +
                                         "' not found in last run results"};
}

const std::unordered_map<std::string, Tensor>& Session::results() const {
  return impl_->last_results;
}

std::unordered_map<std::string, Tensor> Session::take_results() {
  return std::move(impl_->last_results);
}

int Session::num_inputs() const { return impl_->model->executor->num_inputs(); }
int Session::num_outputs() const {
  return impl_->model->executor->num_outputs();
}

}  // namespace omle::rt
