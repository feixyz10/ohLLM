
#include "ohllm/model/model_ctx.h"

#include "ohllm/model/model_loader/model_loader.h"

namespace ohllm::model {

auto ModelCtx::Load(const std::string &path) -> void {
  std::unique_ptr<BaseModelLoader> loader = nullptr;
  if (path.ends_with(".gguf")) { loader = std::make_unique<GGUFModelLoader>(); }
  LOGI_IF(loader == nullptr, "Unsupported model format: %s", path.c_str());
  loader->Load(path, &buffer_, &state_dict_, &config_);
}

}  // namespace ohllm::model