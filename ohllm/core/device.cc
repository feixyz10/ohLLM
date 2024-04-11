#include "ohllm/core/device.h"

#include "ohllm/common/log.h"


namespace ohllm::core {

Device::Device(DeviceType type, I32 id) : type_(type), id_(id) {}

Device::Device(const std::string &name) {
  auto pos = name.find(':');
  std::string type_name = name.substr(0, pos);
  type_ = DEVICE_TYPE_NUMBER;
  for (I32 i = 0; i < DEVICE_TYPE_NUMBER; ++i) {
    if (kDeviceTypeNames[i] == type_name) {
      type_ = static_cast<DeviceType>(i);
      break;
    }
  }
  if (type_ == DEVICE_TYPE_NUMBER) {
    LOGF("Device name (%s) not supported.", name.c_str());
  }
  id_ = pos == std::string::npos ? 0 : std::stoi(name.substr(pos + 1));
}

auto Device::name() const -> std::string {
  return kDeviceTypeNames[type_] + ":" + std::to_string(id_);
}


}  // namespace ohllm::core