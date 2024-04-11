#pragma once

#include "ohllm/core/operator/operator.h"

namespace ohllm::core::cpu {

class OperatorIdentity : public core::BaseOperator {
 public:
  void Run(const core::OpTensorParams &tensor_params,
           const core::OpFloatParams &float_params,
           const core::OpIntParams &int_params) override;
};

class OperatorEmbedding : public core::BaseOperator {
 public:
  void Run(const core::OpTensorParams &tensor_params,
           const core::OpFloatParams &float_params,
           const core::OpIntParams &int_params) override;
};

}  // namespace ohllm::core::cpu