#pragma once

#include "ohllm/common/type.h"

#define BLOCK_SIZE_INT8 32
#define BLOCK_SIZE_INT4 32

namespace ohllm::core {

// note: same to ggml (ggml-quants.h)

template <U64 block_size, bool symmetric>
struct BlockInt8_;

template <U64 block_size>
struct BlockInt8_<block_size, true> {
  FP16 d;
  I8 q[block_size];
};

template <U64 block_size, bool symmetric>
struct BlockInt4_;

template <U64 block_size>
struct BlockInt4_<block_size, true> {
  FP16 d;
  U8 q[block_size / 2];
};

template <U64 block_size>
struct BlockInt4_<block_size, false> {
  FP16 d, m;
  U8 q[block_size / 2];
};

using BlockInt8Sym = BlockInt8_<BLOCK_SIZE_INT8, true>;

using BlockInt4 = BlockInt4_<BLOCK_SIZE_INT4, false>;
using BlockInt4Sym = BlockInt4_<BLOCK_SIZE_INT4, true>;

}  // namespace ohllm::core