#ifndef OMLE_NODES_NEURAL_NETWORK_NODE_H_
#define OMLE_NODES_NEURAL_NETWORK_NODE_H_

#include <memory>
#include <vector>

#include "../graph_node.h"
#include "../model_base.h"

namespace omle::rt::impl {

// Activation enum lives here so factory callers and readers share it.
enum class NNActivation : uint8_t {
  Identity = 0,
  Logistic = 1,
  Tanh = 2,
  Relu = 3,
  Softmax = 4,
};

// Per-layer descriptor passed to factory functions.
// Weights and biases are held as float or double depending on the factory used.
// The layer descriptor itself is always the same struct; the data vectors
// are provided separately to the factory.
struct NNLayerDesc {
  NNActivation activation = NNActivation::Identity;
  int in_features = 0;
  int out_features = 0;
};

// Opaque typed implementation — defined in neural_network_node.cpp only.
class NeuralNetworkNode final : public GraphNode {
 public:
  struct Impl;
  explicit NeuralNetworkNode(std::unique_ptr<Impl> impl);
  ~NeuralNetworkNode();
  omle::rt::Status execute(ValueStore& vs, int n_rows) const override;

 private:
  std::unique_ptr<Impl> impl_;
};

// -----------------------------------------------------------------------
// Per-layer data bundle used by factory functions.
// -----------------------------------------------------------------------
struct NNLayerDataF32 {
  std::vector<float> weights;  // [out_features, in_features] row-major
  std::vector<float> bias;     // [out_features], may be empty
  NNActivation activation = NNActivation::Identity;
  int in_features = 0;
  int out_features = 0;
};

struct NNLayerDataF64 {
  std::vector<double> weights;
  std::vector<double> bias;
  NNActivation activation = NNActivation::Identity;
  int in_features = 0;
  int out_features = 0;
};

// -----------------------------------------------------------------------
// Factory functions (defined in neural_network_node.cpp; no templates leak).
// -----------------------------------------------------------------------

// Build a float32 NeuralNetworkNode::Impl.
std::unique_ptr<NeuralNetworkNode::Impl> make_neural_network_impl(
    std::vector<NNLayerDataF32> layers, int n_features, int n_outputs);

// Build a float64 NeuralNetworkNode::Impl.
std::unique_ptr<NeuralNetworkNode::Impl> make_neural_network_impl_f64(
    std::vector<NNLayerDataF64> layers, int n_features, int n_outputs);

// Node-level factories — return the fully constructed node.
// Use these from TUs where NeuralNetworkNode::Impl is incomplete.
std::unique_ptr<NeuralNetworkNode> make_neural_network_node(
    std::vector<NNLayerDataF32> layers, int n_features, int n_outputs);

std::unique_ptr<NeuralNetworkNode> make_neural_network_node_f64(
    std::vector<NNLayerDataF64> layers, int n_features, int n_outputs);

}  // namespace omle::rt::impl

#endif  // OMLE_NODES_NEURAL_NETWORK_NODE_H_
