#include "ohllm/core/data_type.h"

#include <cstring>
#include <vector>

#include "ohllm/common/log.h"
#include "ohllm/common/type.h"
#include "ohllm/core/quant.h"

namespace ohllm::core {

namespace internal {

void ConvertFp32ToF32Func(F32 *dst, void *src, U64 n) {
  std::memcpy(dst, src, n * sizeof(F32));
}

void ConvertFp16ToF32Func(F32 *dst, void *src, U64 n) {
  FP16 *s = (FP16 *)src;
  for (U64 i = 0; i < n; i++) { dst[i] = CAST_FP16_TO_FP32(s[i]); }
}

void ConvertInt8symToF32Func(F32 *dst, void *src, U64 n) {
  const U64 &bs = BLOCK_SIZE_INT8;
  ASSERT(n % bs == 0, "");
  BlockInt8Sym *source = (BlockInt8Sym *)src;
  for (U64 i = 0; i < n / bs; i++) {
    BlockInt8Sym block = *(source + i);
    F32 d = CAST_FP16_TO_FP32(block.d);
    for (U64 j = 0; j < bs; j++) { dst[i * bs + j] = d * block.q[j]; }
  }
  std::vector<F32> dd(dst, dst + 32);
}

void ConvertInt4ToF32Func(F32 *dst, void *src, U64 n) {
  const U64 &bs = BLOCK_SIZE_INT4;
  ASSERT(n % bs == 0, "");
  BlockInt4 *source = (BlockInt4 *)src;
  for (U64 i = 0; i < n / bs; i++) {
    BlockInt4 block = *(source + i);
    F32 d = CAST_FP16_TO_FP32(block.d);
    F32 m = CAST_FP16_TO_FP32(block.m);
    for (U64 j = 0; j < bs / 2; j++) {
      I32 q0 = block.q[j] & 0x0F;
      I32 q1 = block.q[j] >> 4;
      dst[i * bs + j] = d * q0 + m;
      dst[i * bs + j + bs / 2] = d * q1 + m;
    }
  }
  // std::vector<F32> dd(dst, dst + 32);
}

void ConvertInt4symToF32Func(F32 *dst, void *src, U64 n) {
  const U64 &bs = BLOCK_SIZE_INT4;
  ASSERT(n % bs == 0, "");
  BlockInt4Sym *source = (BlockInt4Sym *)src;
  for (U64 i = 0; i < n / bs; i++) {
    BlockInt4Sym block = *(source + i);
    F32 d = CAST_FP32_TO_FP16(block.d);
    for (U64 j = 0; j < bs / 2; j++) {
      I32 q0 = (block.q[j] & 0x0F) - 8;
      I32 q1 = (block.q[j] >> 4) - 8;
      dst[i * bs + j] = d * q0;
      dst[i * bs + j + bs / 2] = d * q1;
    }
  }
}

}  // namespace internal

}  // namespace ohllm::core
