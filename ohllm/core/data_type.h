#pragma once

#include <functional>
#include <map>
#include <string>

#include "ohllm/common/type.h"
#include "ohllm/core/quant.h"

namespace ohllm::core {

enum DataType {
  DATA_TYPE_FP32 = 0,
  DATA_TYPE_FP16,
  DATA_TYPE_INT8_SYM,  // Q8_0
  DATA_TYPE_INT4,      // Q4_1
  DATA_TYPE_INT4_SYM,  // Q4_0
  DATA_TYPE_NUMBER
};

static const std::map<DataType, std::string> kDataTypeNames = {
    {DATA_TYPE_FP32, "FP32"},      {DATA_TYPE_FP16, "FP16"},
    {DATA_TYPE_INT8_SYM, "INT8S"}, {DATA_TYPE_INT4, "INT4"},
    {DATA_TYPE_INT4_SYM, "INT4S"},
};

static const std::map<std::string, DataType> kDataTypeMap = {
    {"FP32", DATA_TYPE_FP32},      {"FP16", DATA_TYPE_FP16},
    {"INT8S", DATA_TYPE_INT8_SYM}, {"INT4", DATA_TYPE_INT4},
    {"INT4S", DATA_TYPE_INT4_SYM},
};

struct DataTypeTrait {
  DataType dtype;
  std::string name;
  U64 block_size;
  U64 block_nbytes;
  bool is_quantized;

  std::function<void(F32 *, void *, U64)> to_fp32_func;
};

namespace internal {
void ConvertFp32ToF32Func(F32 *dst, void *src, U64 n);

void ConvertFp16ToF32Func(F32 *dst, void *src, U64 n);

void ConvertInt8symToF32Func(F32 *dst, void *src, U64 n);

void ConvertInt4ToF32Func(F32 *dst, void *src, U64 n);

void ConvertInt4symToF32Func(F32 *dst, void *src, U64 n);
}  // namespace internal


static const DataTypeTrait kDataTypeTraits[DataType::DATA_TYPE_NUMBER] = {
    {
        .dtype = DATA_TYPE_FP32,
        .name = kDataTypeNames.at(DATA_TYPE_FP32),
        .block_size = 1,
        .block_nbytes = sizeof(F32),
        .is_quantized = false,
        .to_fp32_func = internal::ConvertFp32ToF32Func,
    },
    {
        .dtype = DATA_TYPE_FP16,
        .name = kDataTypeNames.at(DATA_TYPE_FP16),
        .block_size = 1,
        .block_nbytes = sizeof(FP16),
        .is_quantized = false,
        .to_fp32_func = internal::ConvertFp16ToF32Func,
    },
    {
        .dtype = DATA_TYPE_INT8_SYM,
        .name = kDataTypeNames.at(DATA_TYPE_INT8_SYM),
        .block_size = BLOCK_SIZE_INT8,
        .block_nbytes = sizeof(BlockInt8Sym),
        .is_quantized = true,
        .to_fp32_func = internal::ConvertInt8symToF32Func,
    },
    {
        .dtype = DATA_TYPE_INT4,
        .name = kDataTypeNames.at(DATA_TYPE_INT4),
        .block_size = BLOCK_SIZE_INT4,
        .block_nbytes = sizeof(BlockInt4),
        .is_quantized = true,
        .to_fp32_func = internal::ConvertInt4ToF32Func,
    },
    {
        .dtype = DATA_TYPE_INT4_SYM,
        .name = kDataTypeNames.at(DATA_TYPE_INT4_SYM),
        .block_size = BLOCK_SIZE_INT4,
        .block_nbytes = sizeof(BlockInt4Sym),
        .is_quantized = true,
        .to_fp32_func = internal::ConvertInt4symToF32Func,
    },
};

}  // namespace ohllm::core