#include "neural_network_node.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <type_traits>
#include <vector>

#include "../post_transform.h"
#include "../simd_traits.h"
#include "../width_dispatch.h"

namespace omle::rt::impl {

// -----------------------------------------------------------------------
// Abstract Impl base
// -----------------------------------------------------------------------
struct NeuralNetworkNode::Impl {
  virtual int n_features() const noexcept = 0;
  virtual int n_outputs() const noexcept = 0;
  virtual omle::rt::DataType dtype() const noexcept = 0;
  virtual void compute(const void* features, int n_rows,
                       void* output) const = 0;
  virtual ~Impl() = default;
};

// -----------------------------------------------------------------------
// Typed layer storage
// -----------------------------------------------------------------------
template <typename T>
struct NNLayerT {
  std::vector<T> weights;  // [out_features, in_features] row-major
  std::vector<T> bias;     // [out_features], may be empty
  NNActivation activation = NNActivation::Identity;
  int in_features = 0;
  int out_features = 0;
};

// -----------------------------------------------------------------------
// Typed model storage
// -----------------------------------------------------------------------
template <typename T>
struct NeuralNetworkModelT {
  std::vector<NNLayerT<T>> layers;
  int n_features = 0;
  int n_outputs = 1;
};

// -----------------------------------------------------------------------
// Templated compute kernels
// -----------------------------------------------------------------------
template <typename T>
static void apply_activation_t(T* data, int n, NNActivation act) {
  switch (act) {
    case NNActivation::Identity:
      break;
    case NNActivation::Logistic:
      for (int i = 0; i < n; ++i) data[i] = T(1) / (T(1) + std::exp(-data[i]));
      break;
    case NNActivation::Tanh:
      for (int i = 0; i < n; ++i) data[i] = std::tanh(data[i]);
      break;
    case NNActivation::Relu:
      for (int i = 0; i < n; ++i) data[i] = data[i] > T(0) ? data[i] : T(0);
      break;
    case NNActivation::Softmax: {
      T mx = *std::max_element(data, data + n);
      T sum = T(0);
      for (int i = 0; i < n; ++i) {
        data[i] = std::exp(data[i] - mx);
        sum += data[i];
      }
      const T inv = T(1) / sum;
      for (int i = 0; i < n; ++i) data[i] *= inv;
      break;
    }
  }
}

template <typename T>
static void forward_layer_t(const NNLayerT<T>& layer,
                            const T* input,  // [in_features]
                            T* output)       // [out_features]
{
  const int nin = layer.in_features;
  const int nout = layer.out_features;
  for (int j = 0; j < nout; ++j) {
    const T* w = layer.weights.data() + j * nin;
    T val = layer.bias.empty() ? T(0) : layer.bias[j];
    for (int k = 0; k < nin; ++k) val += w[k] * input[k];
    output[j] = val;
  }
  apply_activation_t<T>(output, nout, layer.activation);
}

// -----------------------------------------------------------------------
// Typed Impl
// -----------------------------------------------------------------------
template <typename T>
struct NeuralNetworkImplT final : NeuralNetworkNode::Impl {
  NeuralNetworkModelT<T> model;

  int n_features() const noexcept override { return model.n_features; }
  int n_outputs() const noexcept override { return model.n_outputs; }
  omle::rt::DataType dtype() const noexcept override {
    return std::is_same<T, double>::value ? omle::rt::DataType::Float64
                                          : omle::rt::DataType::Float32;
  }

  void compute(const void* features_raw, int n_rows,
               void* output_raw) const override {
    const T* features = static_cast<const T*>(features_raw);
    T* output = static_cast<T*>(output_raw);

    // Allocate two ping-pong buffers large enough for the widest layer.
    int max_width = model.n_features;
    for (const auto& l : model.layers)
      max_width = std::max(max_width, l.out_features);

    std::vector<T> buf_a(max_width), buf_b(max_width);
    const int nout = model.n_outputs;

    for (int s = 0; s < n_rows; ++s) {
      const T* x = features + s * model.n_features;

      T* cur = buf_a.data();
      T* nxt = buf_b.data();

      bool first = true;
      for (const auto& layer : model.layers) {
        const T* in_ptr = first ? x : cur;
        forward_layer_t<T>(layer, in_ptr, nxt);
        std::swap(cur, nxt);
        first = false;
      }

      std::memcpy(output + s * nout, cur, nout * sizeof(T));
    }
  }
};

// Explicit instantiations — linker finds these; no other TU instantiates them.
template struct NeuralNetworkImplT<float>;
template struct NeuralNetworkImplT<double>;

// -----------------------------------------------------------------------
// Factory functions
// -----------------------------------------------------------------------
std::unique_ptr<NeuralNetworkNode::Impl> make_neural_network_impl(
    std::vector<NNLayerDataF32> layers, int n_features, int n_outputs) {
  auto p = std::make_unique<NeuralNetworkImplT<float>>();
  p->model.n_features = n_features;
  p->model.n_outputs = n_outputs;
  p->model.layers.reserve(layers.size());
  for (auto& ld : layers) {
    NNLayerT<float> layer;
    layer.weights = std::move(ld.weights);
    layer.bias = std::move(ld.bias);
    layer.activation = ld.activation;
    layer.in_features = ld.in_features;
    layer.out_features = ld.out_features;
    p->model.layers.push_back(std::move(layer));
  }
  return p;
}

std::unique_ptr<NeuralNetworkNode::Impl> make_neural_network_impl_f64(
    std::vector<NNLayerDataF64> layers, int n_features, int n_outputs) {
  auto p = std::make_unique<NeuralNetworkImplT<double>>();
  p->model.n_features = n_features;
  p->model.n_outputs = n_outputs;
  p->model.layers.reserve(layers.size());
  for (auto& ld : layers) {
    NNLayerT<double> layer;
    layer.weights = std::move(ld.weights);
    layer.bias = std::move(ld.bias);
    layer.activation = ld.activation;
    layer.in_features = ld.in_features;
    layer.out_features = ld.out_features;
    p->model.layers.push_back(std::move(layer));
  }
  return p;
}

// Node-level wrappers — defined here where Impl is complete.
std::unique_ptr<NeuralNetworkNode> make_neural_network_node(
    std::vector<NNLayerDataF32> layers, int n_features, int n_outputs) {
  return std::make_unique<NeuralNetworkNode>(
      make_neural_network_impl(std::move(layers), n_features, n_outputs));
}

std::unique_ptr<NeuralNetworkNode> make_neural_network_node_f64(
    std::vector<NNLayerDataF64> layers, int n_features, int n_outputs) {
  return std::make_unique<NeuralNetworkNode>(
      make_neural_network_impl_f64(std::move(layers), n_features, n_outputs));
}

// -----------------------------------------------------------------------
// NeuralNetworkNode
// -----------------------------------------------------------------------
NeuralNetworkNode::NeuralNetworkNode(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
NeuralNetworkNode::~NeuralNetworkNode() = default;

omle::rt::Status NeuralNetworkNode::execute(ValueStore& vs, int n_rows) const {
  const omle::rt::DataType dt = impl_->dtype();
  auto features = gather_slots(vs, in_names, n_rows, dt);

  const int n_out = impl_->n_outputs();
  const bool f64 = (dt == omle::rt::DataType::Float64);
  omle::rt::Tensor scores = omle::rt::Tensor::dense(dt, n_rows, n_out);
  with_width(f64, [&](auto tag) {
    using T = decltype(tag);
    impl_->compute((const void*)data_w<T>(features), n_rows,
                   (void*)data_w<T>(scores));
  });

  // Classification: two output names → [y_pred, y_prob].
  // y_prob: full probability matrix; y_pred: class label or threshold.
  if (out_names.size() == 2) {
    omle::rt::Tensor pred =
        omle::rt::Tensor::dense(omle::rt::DataType::Int64, n_rows, 1);
    int64_t* pred_ptr = pred.i64_ptr();
    if (n_out == 1) {
      // Binary: expand (n,1) → (n,2): [1-p, p]
      omle::rt::Tensor prob2 = omle::rt::Tensor::dense(dt, n_rows, 2);
      with_width(f64, [&](auto tag) {
        using T = decltype(tag);
        const T* sp = data_w<T>(scores);
        T* dp = data_w<T>(prob2);
        for (int r = 0; r < n_rows; ++r) {
          dp[r * 2] = T(1) - sp[r];
          dp[r * 2 + 1] = sp[r];
          pred_ptr[r] = sp[r] >= T(0.5) ? 1 : 0;
        }
      });
      vs.put(out_names[1], std::move(prob2));
    } else {
      omle::rt::Tensor prob = scores;
      vs.put(out_names[1], std::move(prob));
      with_width(f64, [&](auto tag) {
        using T = decltype(tag);
        const T* sp = data_w<T>(scores);
        for (int r = 0; r < n_rows; ++r) {
          const T* row = sp + r * n_out;
          pred_ptr[r] = static_cast<int64_t>(
              std::distance(row, std::max_element(row, row + n_out)));
        }
      });
    }
    vs.put(out_names[0], std::move(pred));
  } else {
    with_width(f64, [&](auto tag) {
      using T = decltype(tag);
      scatter_outputs(vs, out_names, data_w<T>(scores), n_rows, n_out);
    });
  }
  return {};
}

}  // namespace omle::rt::impl
