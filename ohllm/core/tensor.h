#pragma once

#include <numeric>
#include <string>
#include <vector>

#include "ohllm/common/log.h"
#include "ohllm/core/data_type.h"

namespace ohllm::core {

class Tensor {
 public:
  Tensor() = default;

  explicit Tensor(const std::vector<I32> &shape, void *data = nullptr,
                  DataType dtype = DATA_TYPE_FP32,
                  const std::string &name = "unknown")
      : data_(data),
        shape_(shape),
        dtype_(dtype),
        dtype_trait_(kDataTypeTraits[dtype]),
        name_(name) {
    InitStride();
    if (data_ == nullptr && nbytes() > 0) {
      owned_ = true;
      data_ = malloc(nbytes());
    }
  }

  ~Tensor() {
    if (owned_) { free(data_); }
  }

  Tensor &operator=(const Tensor &t) {
    data_ = t.data_;
    owned_ = false;
    shape_ = t.shape_;
    stride_ = t.stride_;
    dtype_ = t.dtype_;
    dtype_trait_ = t.dtype_trait_;
    name_ = t.name_;
    return *this;
  }

  Tensor(const Tensor &t) : Tensor(t.shape_, t.data_, t.dtype_, t.name_){};

  Tensor(Tensor &&t) = delete;

  Tensor &operator=(Tensor &&t) = delete;

  auto name() const -> const std::string & { return name_; }

  auto dtype() const -> DataType { return dtype_; }

  auto dtype_trait() const -> const DataTypeTrait & { return dtype_trait_; }

  auto shape() const -> const std::vector<I32> & { return shape_; }

  auto stride() const -> const std::vector<I32> & { return stride_; }

  auto stride_bytes() const -> const std::vector<I32> & {
    return stride_bytes_;
  }

  auto ndim() const -> U64 { return shape_.size(); }

  auto numel() const -> U64 {
    return std::accumulate(shape_.begin(), shape_.end(), 1,
                           std::multiplies<I32>());
  }

  auto nbytes() const -> U64 {
    return numel() / dtype_trait_.block_size * dtype_trait_.block_nbytes;
  }

  auto data() const -> void * { return data_; }

  auto owned() const -> bool { return owned_; }

  auto is_contiguous() const -> bool;

  // auto view(std::vector<I32> shape) const -> Tensor;

  // auto permute(std::vector<I32> perm) const -> Tensor;

  // auto transpose(I32 axis1, I32 axis2) const -> Tensor;

  // auto operator()(const std::vector<I32> &index) const -> Tensor;

  auto ToString() const -> std::string;

  auto ToF32() -> std::vector<F32>;

 private:
  void InitStride();

  void *data_ = nullptr;
  bool owned_ = false;
  std::vector<I32> shape_;
  std::vector<I32> stride_;
  std::vector<I32> stride_bytes_;
  DataType dtype_ = DATA_TYPE_FP32;
  DataTypeTrait dtype_trait_;
  std::string name_ = "unknown";
};

}  // namespace ohllm::core

inline std::ostream &operator<<(std::ostream &out,
                                const ::ohllm::core::Tensor &t) {
  out << t.ToString();
  return out;
}