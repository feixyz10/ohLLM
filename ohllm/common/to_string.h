#pragma once

#include <map>
#include <ostream>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace ohllm::common {

template <typename T>
std::string ToString(const std::vector<T> &v) {
  std::ostringstream oss;
  oss << "[";
  for (size_t i = 0; i < v.size(); ++i) {
    oss << v[i];
    if (i != v.size() - 1) oss << ", ";
  }
  oss << "]";
  return oss.str();
}

template <typename T1, typename T2>
std::string ToString(std::unordered_map<T1, T2> &m) {
  std::ostringstream oss;
  oss << "{";
  size_t i = 0;
  for (const auto &[k, v] : m) {
    oss << k << ": " << v;
    if (i++ != m.size() - 1) oss << ", ";
  }
  oss << "}";
  return oss.str();
}

template <typename T1, typename T2>
std::string ToString(std::map<T1, T2> &m) {
  std::ostringstream oss;
  oss << "{";
  size_t i = 0;
  for (const auto &[k, v] : m) {
    oss << k << ": " << v;
    if (i++ != m.size() - 1) oss << ", ";
  }
  oss << "}";
  return oss.str();
}

template <typename T1, typename T2>
std::string ToString(std::set<T1, T2> &m) {
  std::ostringstream oss;
  oss << "{";
  size_t i = 0;
  for (const auto &k : m) {
    oss << k;
    if (i++ != m.size() - 1) oss << ", ";
  }
  oss << "}";
  return oss.str();
}

template <typename T1, typename T2>
std::string ToString(std::unordered_set<T1, T2> &m) {
  std::ostringstream oss;
  oss << "{";
  size_t i = 0;
  for (const auto &k : m) {
    oss << k;
    if (i++ != m.size() - 1) oss << ", ";
  }
  oss << "}";
  return oss.str();
}

inline std::string ToString() { return ""; }

template <typename... T>
std::string ToString() {
  return "";
}

template <typename T, typename... Ts>
std::string ToString(T t, Ts... ts) {
  std::ostringstream oss;
  oss << t;
  if constexpr (sizeof...(ts) > 0) oss << ", ";
  oss << ToString(ts...);
  return oss.str();
}

}  // namespace ohllm::common

template <typename T>
std::ostream &operator<<(std::ostream &out, const std::vector<T> &v) {
  out << ::ohllm::common::ToString(v);
  return out;
}