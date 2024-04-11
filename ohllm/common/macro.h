#pragma once

#include <cmath>
#include <numeric>

#include "type.h"

#define UNUSED(x) (void)(x)

#define MAX_INT32 std::numeric_limits<I32>::max()
#define MIN_INT32 std::numeric_limits<I32>::min()
#define MAX_UINT32 std::numeric_limits<U32>::max()

#define MIN(a, b) ((a) < (b) ? (a) : (b))

#define MAX(a, b) ((a) > (b) ? (a) : (b))

#define ABS(x) ((x) > 0 ? (x) : -(x))

#define ROUND(x) ((x) > 0 ? (I32)((x) + 0.5) : (I32)((x)-0.5))

#define FLOOR(x)                                                       \
  ({                                                                   \
    I32 _ret = (I32)x;                                                 \
    if (ROUND(x) != (x)) { _ret = (x) > 0 ? (I32)(x) : (I32)((x)-1); } \
    _ret;                                                              \
  })

#define CEIL(x)                                                        \
  ({                                                                   \
    I32 _ret = (I32)x;                                                 \
    if (ROUND(x) != (x)) { _ret = (x) > 0 ? (I32)((x) + 1) : (I32)x; } \
    _ret;                                                              \
  })

#define CLAMP(x, min, max) (MIN(MAX(x, min), max))

#define SWAP(a, b) \
  {                \
    typeof(a) t;   \
    t = a;         \
    a = b;         \
    b = t;         \
  }

#define NEAR(x, y, epsilon) (ABS((x) - (y)) < (epsilon))

#define NEAR_ZERO(x, epsilon) (ABS(x) < (epsilon))

#define VSIZE(v) static_cast<I32>((v).size())

#define VSIZEI32(v) static_cast<I32>((v).size())

#define VSIZEU32(v) static_cast<U32>((v).size())

#define VSIZEI64(v) static_cast<I64>((v).size())

#define VSIZEU64(v) static_cast<U64>((v).size())

#define EPSILON_F32 std::numeric_limits<F32>::epsilon();

#define EPSILON_F64 std::numeric_limits<F64>::epsilon();

#define DISALLOW_COPY_AND_ASSIGN(classname) \
  classname(const classname &) = delete;    \
  classname &operator=(const classname &) = delete; /* NOLINT */

#define DISALLOW_INSTANTIATION(classname) \
  classname() = delete;                   \
  classname(const classname &) = delete;  \
  classname &operator=(const classname &) = delete; /* NOLINT */

#define DECLARE_SINGLETON(classname)          \
 public:                                      \
  static classname *Instance() { /* NOLINT */ \
    static classname instance;                \
    return &instance;                         \
  }                                           \
                                              \
 private:                                     \
  classname();                                \
  DISALLOW_COPY_AND_ASSIGN(classname)

// NOLINTNEXTLINE
#define DECLARE_SINGLETON_WITH_DEFAULT_CONSTRUCTOR(classname) \
 public:                                                      \
  static classname *Instance() { /* NOLINT */                 \
    static classname instance;                                \
    return &instance;                                         \
  }                                                           \
                                                              \
 private:                                                     \
  classname() = default;                                      \
  DISALLOW_COPY_AND_ASSIGN(classname)
