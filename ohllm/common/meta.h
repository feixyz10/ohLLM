#pragma once

#include "ohllm/common/type.h"

namespace ohllm::common {

template <typename T>
concept Arithmetic = std::is_arithmetic_v<T>;

template <typename T>
concept Integral = std::is_integral_v<T>;

template <typename T>
concept FloatingPoint = std::is_floating_point_v<T>;

template <typename T>
concept Enum = std::is_enum_v<T>;

template <typename T>
concept ArithmeticOrEnum = Arithmetic<T> || Enum<T>;

}  // namespace ohllm::common