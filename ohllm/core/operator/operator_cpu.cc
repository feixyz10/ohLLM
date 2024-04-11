#include "ohllm/core/operator/operator_cpu.h"

#include "ohllm/common/macro.h"

namespace ohllm::core::cpu {

void OperatorIdentity::Run(const core::OpTensorParams &tensor_params,
                           const core::OpFloatParams &float_params,
                           const core::OpIntParams &int_params) {
  UNUSED(tensor_params);
  UNUSED(float_params);
  UNUSED(int_params);
}

void OperatorEmbedding ::Run(const core::OpTensorParams &tensor_params,
                             const core::OpFloatParams &float_params,
                             const core::OpIntParams &int_params) {
  UNUSED(tensor_params);
  UNUSED(float_params);
  UNUSED(int_params);
}

}  // namespace ohllm::core::cpu