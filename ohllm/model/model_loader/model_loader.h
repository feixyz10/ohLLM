#pragma once

#include <any>
#include <map>
#include <string>

#include "ohllm/core/tensor.h"
#include "ohllm/model/model_config.h"
#include "ohllm/model/state_dict.h"

namespace ohllm::model {

enum ModelLoaderType {
  MODEL_LOADER_TYPE_GGUF = 0,
  MODEL_LOADER_TYPE_OHLLM,
  MODEL_LOADER_TYPE_NUMBER
};

class BaseModelLoader {
 public:
  explicit BaseModelLoader(ModelLoaderType type) : type_(type) {}

  BaseModelLoader(const BaseModelLoader &) = delete;

  BaseModelLoader &operator=(const BaseModelLoader &) = delete;

  virtual ~BaseModelLoader() = default;

  virtual void Load(const std::string &filename, std::vector<char> *buffer,
                    StateDict *state_dict, ModelConfig *config = nullptr) = 0;

 protected:
  ModelLoaderType type_;
};

class GGUFModelLoader : public BaseModelLoader {
 public:
  GGUFModelLoader() : BaseModelLoader(MODEL_LOADER_TYPE_GGUF) {}

  void Load(const std::string &filename, std::vector<char> *buffer,
            StateDict *state_dict,
            ModelConfig *config = nullptr) override final;
};

}  // namespace ohllm::model