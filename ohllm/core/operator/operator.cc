#include "ohllm/core/operator/operator.h"

#include "ohllm/common/log.h"
#include "ohllm/common/type.h"
#include "ohllm/core/operator/operator_cpu.h"

namespace ohllm::core {

OperatorManager::OperatorManager(core::DeviceType device) : device_(device) {
  if (device_ == DEVICE_TYPE_CPU) {
    ops_[OP_TYPE_IDENTITY] = std::make_unique<cpu::OperatorIdentity>();
    ops_[OP_TYPE_EMBEDDING] = std::make_unique<cpu::OperatorEmbedding>();
  } else {
    LOGF("Device (%s) not supported.", kDeviceTypeNames[device].c_str());
  }
}

void OperatorManager::Run(OperatorType op_type,
                          const OpTensorParams &tensor_params,
                          const OpFloatParams &float_params,
                          const OpIntParams &int_params) {
  auto iter = ops_.find(op_type);
  if (iter == ops_.end()) {
    LOGF("Operator (%s) not supported.", kOperatorTypeNames[op_type].c_str());
  }
  iter->second->Run(tensor_params, float_params, int_params);
}


}  // namespace ohllm::core