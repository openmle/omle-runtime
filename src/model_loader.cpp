#include "model_loader.h"

#include <fstream>
#include <string>

#include "graph_loader.h"

namespace omle::rt::impl {

LoadedModel load_from_bytes(const void* data, std::size_t size,
                            const LoadOptions& opts) {
  return build_graph_executor(data, size, opts);
}

LoadedModel load_from_file(const std::string& path, const LoadOptions& opts) {
  std::ifstream ifs(path, std::ios::binary | std::ios::ate);
  if (!ifs) throw std::runtime_error("omle: cannot open model file: " + path);
  const auto sz = ifs.tellg();
  ifs.seekg(0);
  std::string buf(static_cast<std::size_t>(sz), '\0');
  if (!ifs.read(buf.data(), sz))
    throw std::runtime_error("omle: failed to read model file: " + path);
  return load_from_bytes(buf.data(), buf.size(), opts);
}

}  // namespace omle::rt::impl
