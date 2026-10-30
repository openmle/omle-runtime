#ifndef OMLE_MODEL_LOADER_H_
#define OMLE_MODEL_LOADER_H_

#include <memory>
#include <string>
#include <vector>

#include "model_base.h"
#include "omle/runtime.h"  // TensorInfo, OutputInfo, LoadOptions

namespace omle::rt::impl {

// Bundle returned by the loader: executor + input/output metadata.
struct LoadedModel {
  std::unique_ptr<ModelBase> executor;
  std::vector<InputSpec> inputs;
  std::vector<OutputSpec> outputs;
  // Non-fatal advisories raised while loading, for the caller to surface.
  // Collected rather than printed: this is a library, and an embedder cannot
  // intercept, route or suppress a write to stderr.
  std::vector<std::string> warnings;
};

// Parse a serialised OMLE protobuf from a file.
// Throws std::runtime_error on failure.
LoadedModel load_from_file(const std::string& path,
                           const LoadOptions& opts = {});

// Parse a serialised OMLE protobuf from a memory buffer.
// Throws std::runtime_error on failure.
LoadedModel load_from_bytes(const void* data, std::size_t size,
                            const LoadOptions& opts = {});

}  // namespace omle::rt::impl

#endif  // OMLE_MODEL_LOADER_H_
