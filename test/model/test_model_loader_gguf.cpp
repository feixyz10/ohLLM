#include <vector>

#include "ohllm/common/log.h"
#include "ohllm/common/to_string.h"
#include "ohllm/model/model_ctx.h"

using namespace ::ohllm::model;
using namespace ::ohllm::core;
using namespace ::ohllm::common;


int main(int argc, char **argv) {
  // std::string model_name = ".temp/gguf/qwen1_5-0_5b-chat-q4_0.gguf";
  std::string model_name = ".temp/Qwen1.5-0.5B-Chat_q8_0.gguf";
  if (argc == 2) { model_name = argv[1]; }
  LOGI("loading model: %s", model_name.c_str());
  ModelCtx::Instance()->Load(model_name);
  for (const auto &sd : ModelCtx::Instance()->state_dict()) {
    std::string name = sd.first;
    Tensor tensor = sd.second;
    auto data_f32 = tensor.ToF32();
    LOGI("%s: %s", name.c_str(), tensor.ToString().c_str());
  }
  return 0;
}