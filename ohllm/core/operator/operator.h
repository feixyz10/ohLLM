#pragma once

#include <map>
#include <string>

#include "ohllm/common/type.h"
#include "ohllm/core/device.h"
#include "ohllm/core/tensor.h"

namespace ohllm::core {

enum OperatorType {
  OP_TYPE_IDENTITY = 0,
  OP_TYPE_EMBEDDING,
  OP_TYPE_LAYER_NORM,
  OP_TYPE_RMS_NORM,
  OP_TYPE_LINEAR,
  OP_TYPE_ROPE,
  OP_TYPE_MHA_ATTN,    // multi-head-attention
  OP_TYPE_GQA_ATTN,    // grouped query attention
  OP_TYPE_FLASH_ATTN,  // flash attention
  OP_TYPE_GELU,
  OP_TYPE_SILU,
  OP_TYPE_RELU,
  OP_TYPE_SOFTMAX,
  OP_TYPE_ARGMAX,
  OP_TYPE_NUMBER
};

static const std::string kOperatorTypeNames[] = {
    "indentity", "embedding", "layer_norm", "rms_norm", "linear",
    "rope",      "attn_mha",  "attn_gqa",   "gelu",     "silu",
    "relu",      "softmax",   "argmax"};

using OpTensorParams = std::map<OperatorType, core::Tensor *>;
using OpFloatParams = std::map<OperatorType, F32>;
using OpIntParams = std::map<OperatorType, I32>;

class BaseOperator {
 public:
  virtual ~BaseOperator() = default;
  virtual void Run(const OpTensorParams &tensor_params,
                   const OpFloatParams &float_params,
                   const OpIntParams &int_params) = 0;
};

class OperatorManager {
 public:
  explicit OperatorManager(core::DeviceType device = DEVICE_TYPE_CPU);

  void Run(OperatorType op_type, const OpTensorParams &tensor_params,
           const OpFloatParams &float_params, const OpIntParams &int_params);

 protected:
  std::map<OperatorType, std::unique_ptr<BaseOperator>> ops_;

  core::DeviceType device_;
};


}  // namespace ohllm::core