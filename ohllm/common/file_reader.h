#pragma once

#include <fstream>
#include <string>
#include <type_traits>
#include <vector>

#include "ohllm/common/meta.h"
#include "ohllm/common/type.h"

namespace ohllm::common {

class BinaryFileReader {
 public:
  explicit BinaryFileReader(const std::string &filename) {
    fin_ = std::make_unique<std::ifstream>(filename, std::ios::binary);
  }

  bool good() { return fin_->good(); }

  I32 tellg() { return fin_->tellg(); }

  void seekg(I32 offset) { fin_->seekg(offset); }

  void ReadBytes(char *data, U64 size) { fin_->read(data, size); }

  template <ArithmeticOrEnum T>
  T Read() {
    T data;
    fin_->read((char *)&data, sizeof(T));
    return data;
  }

  auto ReadString(U64 n) -> std::string {
    std::string str(n, '\0');
    ReadBytes(str.data(), static_cast<U64>(n));
    return str;
  }

  template <Arithmetic T>
  auto ReadVector(U64 n) -> std::vector<T> {
    std::vector<T> vec(n);
    ReadBytes((char *)vec.data(), static_cast<U64>(n * sizeof(T)));
    return vec;
  }

  template <>
  auto ReadVector<bool>(U64 n) -> std::vector<bool> {
    std::vector<bool> vec(n);
    for (U64 i = 0; i < n; ++i) { vec[i] = Read<bool>(); }
    return vec;
  }

 private:
  std::unique_ptr<std::ifstream> fin_;
};

}  // namespace ohllm::common