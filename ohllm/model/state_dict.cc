#include "ohllm/model/state_dict.h"

#include "ohllm/common/log.h"
#include "ohllm/model/model_loader/model_loader.h"

namespace ohllm::model {

auto StateDict::Names() const -> std::vector<std::string> {
  std::vector<std::string> keys;
  for (const auto &item : state_dict_) { keys.push_back(item.first); }
  return keys;
}

auto StateDict::Exists(const std::string &key) const -> bool {
  return state_dict_.find(key) != state_dict_.end();
}

auto StateDict::operator[](const std::string &key) const
    -> ::ohllm::core::Tensor {
  auto iter = state_dict_.find(key);
  if (iter == state_dict_.end()) { LOGF("State (%s) not found.", key.c_str()); }
  return iter->second;
}

auto StateDict::AddTensor(const std::string &key,
                          const ::ohllm::core::Tensor &tensor) -> void {
  state_dict_.insert({key, tensor});
}

}  // namespace ohllm::model