#pragma once

#include <any>
#include <map>
#include <string>
#include <vector>

#include "ohllm/core/tensor.h"

namespace ohllm::model {


class StateDict {
 private:
  using States = std::map<std::string, ::ohllm::core::Tensor>;

 public:
  auto Names() const -> std::vector<std::string>;

  auto Exists(const std::string &key) const -> bool;

  auto operator[](const std::string &key) const -> ::ohllm::core::Tensor;

  auto AddTensor(const std::string &key, const ::ohllm::core::Tensor &tensor)
      -> void;

  auto Clear() -> void { state_dict_.clear(); }

  auto begin() const -> States::const_iterator { return state_dict_.cbegin(); }

  auto end() const -> States::const_iterator { return state_dict_.cend(); }

  auto size() const -> U64 { return state_dict_.size(); }

 private:
  States state_dict_;
};

}  // namespace ohllm::model