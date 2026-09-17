#ifndef OMLE_GRAPH_NODE_H_
#define OMLE_GRAPH_NODE_H_

#include <string>
#include <vector>

#include "omle/status.h"
#include "runtime_tensor.h"
#include "thread_pool.h"

namespace omle::rt::impl {

// Abstract base for all graph nodes.
// A node reads from and writes to a shared ValueStore.
class GraphNode {
 public:
  virtual ~GraphNode() = default;

  // Names of values this node expects to find in the ValueStore.
  std::vector<std::string> in_names;

  // Names of values this node will write into the ValueStore.
  std::vector<std::string> out_names;

  // Symbolic node name (for error messages).
  std::string node_name;

  // Execute computation; read from vs[in_names], write results to
  // vs[out_names].
  virtual omle::rt::Status execute(ValueStore& vs, int n_rows) const = 0;

  // Optional: nodes that support intra-call parallelism override this.
  virtual void set_thread_pool(ThreadPool* /*pool*/) {}
};

}  // namespace omle::rt::impl

#endif  // OMLE_GRAPH_NODE_H_
