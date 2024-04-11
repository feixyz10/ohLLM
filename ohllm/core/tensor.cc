#include "ohllm/core/tensor.h"

#include <sstream>
#include <vector>

#include "ohllm/common/log.h"
#include "ohllm/common/to_string.h"

namespace ohllm::core {

namespace {
auto CalcStride(Tensor *t) -> std::vector<I32> {
  const auto &shape = t->shape();
  I32 nd = shape.size();
  if (nd == 0) { return {}; }
  std::vector<I32> stride(nd, 0);
  stride[nd - 1] = 1;
  for (I32 i = nd - 2; i >= 0; --i) {
    stride[i] = stride[i + 1] * shape[i + 1];
  }
  return stride;
}

auto CalcStrideBytes(Tensor *t) -> std::vector<I32> {
  const auto &shape = t->shape();
  const auto &dtype_trait = t->dtype_trait();
  I32 nd = t->shape().size();
  if (nd == 0) { return {}; }
  std::vector<I32> stride(nd);
  stride[nd - 1] = dtype_trait.block_nbytes;
  if (nd == 1) { return stride; }
  stride[nd - 2] = stride[nd - 1] * shape[nd - 1] / dtype_trait.block_size;
  for (I32 i = nd - 3; i >= 0; --i) {
    stride[i] = stride[i + 1] * shape[i + 1];
  }
  return stride;
}
}  // namespace

void Tensor::InitStride() {
  stride_ = CalcStride(this);
  stride_bytes_ = CalcStrideBytes(this);
}

auto Tensor::is_contiguous() const -> bool {
  I32 s = 1, d = ndim() - 1;
  while (d >= 0) {
    if (stride_[d] != s) { return false; }
    s *= shape_[d--];
  }
  return true;
}

// auto Tensor::view(std::vector<I32> shape) const -> Tensor {
//   ASSERT(is_contiguous(), "Tensor is not contiguous");
// }

// auto Tensor::permute(std::vector<I32> perm) const -> Tensor {}

// auto Tensor::operator()(std::vector<I32> index) const -> Tensor {}

auto Tensor::ToString() const -> std::string {
  std::ostringstream oss;
  oss << "Tensor(shape=" << shape() << ", dtype=" << dtype_trait_.name
      << ", stride=" << stride() << ")" << std::endl;
  return oss.str();
}

auto Tensor::ToF32() -> std::vector<F32> {
  U64 n = numel(), bs = dtype_trait_.block_size;
  ASSERT(n % bs == 0, "");
  std::vector<F32> ret(n);
  dtype_trait_.to_fp32_func(ret.data(), data_, n);
  return ret;
}


}  // namespace ohllm::core