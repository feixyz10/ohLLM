#pragma once

#include <cstdint>
#include <string>

using I8 = int8_t;
using I16 = int16_t;
using I32 = int32_t;
using I64 = int64_t;

using U8 = uint8_t;
using U16 = uint16_t;
using U32 = uint32_t;
using U64 = uint64_t;

using FP16 = __fp16;
using F32 = float;
using FP32 = float;
using F64 = double;
using FP64 = double;

#define CAST_FP16_TO_FP32(x) (FP32)(x)

#define CAST_FP32_TO_FP16(x) (FP16)(x)