#pragma once

#include <any>
#include <vector>

#include "ohllm/common/macro.h"
#include "ohllm/core/tensor.h"
#include "ohllm/model/model_config.h"
#include "ohllm/model/state_dict.h"

namespace ohllm::model {

class ModelCtx {
  DECLARE_SINGLETON_WITH_DEFAULT_CONSTRUCTOR(ModelCtx)

 public:
  auto Init(U64 n) -> void { buffer_.resize(n); }

  auto weight() const -> const std::vector<char> & { return buffer_; }

  auto weight_ptr() -> char * { return buffer_.data(); }

  auto WeightNbytes() const -> U64 { return buffer_.size(); }

  auto config() const -> const ModelConfig & { return config_; }

  auto config_ptr() -> ModelConfig * { return &config_; }

  auto state_dict() const -> const StateDict & { return state_dict_; }

  auto state_dict_ptr() -> StateDict * { return &state_dict_; }

  auto Tensor(const std::string &name) const -> ::ohllm::core::Tensor {
    return state_dict_[name];
  }

  auto AddTensor(const std::string &name, ::ohllm::core::Tensor tensor)
      -> void {
    state_dict_.AddTensor(name, tensor);
  }

  auto Load(const std::string &path) -> void;

  auto Clear() -> void {
    buffer_.clear();
    state_dict_.Clear();
  }

 public:
 private:
  std::vector<char> buffer_;
  ModelConfig config_;
  StateDict state_dict_;
};

}  // namespace ohllm::model