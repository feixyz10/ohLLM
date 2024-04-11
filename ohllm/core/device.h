#pragma once

#include <string>

#include "ohllm/common/type.h"

namespace ohllm::core {

enum DeviceType { DEVICE_TYPE_CPU = 0, DEVICE_TYPE_GPU, DEVICE_TYPE_NUMBER };

static const std::string kDeviceTypeNames[] = {"cpu", "gpu"};

class Device {
 public:
  Device(DeviceType type = DEVICE_TYPE_CPU, I32 id = 0);

  Device(const std::string &str);

  auto name() const -> std::string;

 private:
  DeviceType type_;
  I32 id_;
};

}  // namespace ohllm::core