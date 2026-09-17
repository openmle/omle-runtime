#ifndef OMLE_GRAPH_LOADER_H_
#define OMLE_GRAPH_LOADER_H_

#include "model_loader.h"
#include "omle/runtime.h"

namespace omle::rt::impl {

// Build a GraphExecutor from the proto model, supporting all node types.
// This is the only new file that includes omle.pb.h.
LoadedModel build_graph_executor(const void* data, std::size_t size,
                                 const LoadOptions& opts = {});

}  // namespace omle::rt::impl

#endif  // OMLE_GRAPH_LOADER_H_
