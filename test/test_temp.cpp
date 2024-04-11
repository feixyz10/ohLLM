#include <algorithm>
#include <any>
#include <iostream>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include "ohllm/common/to_string.h"
#include "ohllm/common/type.h"

int main() {
  std::any a = 1;
  int b = std::any_cast<int>(a);
  std::cout << b << std::endl;

  a = std::string("hello");
  std::string s = std::any_cast<std::string>(a);
  std::cout << s << std::endl;

  char x[] = {0, 1, 0, 1};
  I16 *ptr = reinterpret_cast<I16 *>(x);
  std::vector<short> y(ptr, ptr + sizeof(x) / 2);
  std::cout << y << std::endl;

  auto data = std::unique_ptr<int>(new int[10]);
  std::fill(data.get(), data.get() + 10, 10);
  std::cout << std::vector<int>(data.get(), data.get() + 10) << std::endl;

  std::cout << std::is_integral_v<U8> << std::is_integral_v<I8> << std::endl;
  std::cout << std::is_integral_v<U16> << std::is_integral_v<I16> << std::endl;
  std::cout << std::is_integral_v<U32> << std::is_integral_v<I32> << std::endl;
  std::cout << std::is_integral_v<U64> << std::is_integral_v<I64> << std::endl;
  std::cout
      << std::is_floating_point_v<
             FP16> << std::is_floating_point_v<F32> << std::is_floating_point_v<F64> << std::endl;
  return 0;
}