#include <any>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "ohllm/common/file_reader.h"
#include "ohllm/common/log.h"
#include "ohllm/core/data_type.h"
#include "ohllm/core/tensor.h"
#include "ohllm/model/model_ctx.h"
#include "ohllm/model/model_loader/model_loader.h"

namespace ohllm::model {

#define GGUF_MAGIC std::string("GGUF")
#define GGML_MAX_DIMS 4
#define GGUF_DEFAULT_ALIGNMENT 32
#define GGML_PAD(x, n) (((x) + (n)-1) & ~((n)-1))

namespace internal {

enum GGUFType {
  GGUF_TYPE_UINT8 = 0,
  GGUF_TYPE_INT8 = 1,
  GGUF_TYPE_UINT16 = 2,
  GGUF_TYPE_INT16 = 3,
  GGUF_TYPE_UINT32 = 4,
  GGUF_TYPE_INT32 = 5,
  GGUF_TYPE_FLOAT32 = 6,
  GGUF_TYPE_BOOL = 7,
  GGUF_TYPE_STRING = 8,
  GGUF_TYPE_ARRAY = 9,
  GGUF_TYPE_UINT64 = 10,
  GGUF_TYPE_INT64 = 11,
  GGUF_TYPE_FLOAT64 = 12,
  GGUF_TYPE_COUNT,
};

const std::vector<std::string> kGGUFTypeNames = {
    "uint8", "int8",   "uint16", "int16",  "uint32", "int32",  "float32",
    "bool",  "string", "array",  "uint64", "int64",  "float64"};

enum GGMLType {
  GGML_TYPE_F32 = 0,
  GGML_TYPE_F16 = 1,
  GGML_TYPE_Q4_0 = 2,
  GGML_TYPE_Q4_1 = 3,
  GGML_TYPE_Q5_0 = 6,
  GGML_TYPE_Q5_1 = 7,
  GGML_TYPE_Q8_0 = 8,
  GGML_TYPE_Q8_1 = 9,
  GGML_TYPE_Q2_K = 10,
  GGML_TYPE_Q3_K = 11,
  GGML_TYPE_Q4_K = 12,
  GGML_TYPE_Q5_K = 13,
  GGML_TYPE_Q6_K = 14,
  GGML_TYPE_Q8_K = 15,
  GGML_TYPE_IQ2_XXS = 16,
  GGML_TYPE_IQ2_XS = 17,
  GGML_TYPE_I8,
  GGML_TYPE_I16,
  GGML_TYPE_I32,
  GGML_TYPE_COUNT,
};

struct GGMLTypeInfo {
  U32 block_size;
  U32 nbytes;
  std::string name;
  core::DataType dtype;
};

const std::map<GGMLType, GGMLTypeInfo> kGGMLTypeInfos = {
    {GGML_TYPE_F32, {1, 4, "f32", core::DATA_TYPE_FP32}},
    {GGML_TYPE_F16, {1, 2, "f16", core::DATA_TYPE_FP16}},
    {GGML_TYPE_Q4_0, {32, 18, "q4_0", core::DATA_TYPE_INT4_SYM}},
    {GGML_TYPE_Q4_1, {32, 20, "q4_1", core::DATA_TYPE_INT4}},
    {GGML_TYPE_Q8_0, {32, 34, "q8_0", core::DATA_TYPE_INT8_SYM}},
};

std::any ReadGGUF(::ohllm::common::BinaryFileReader &freader, GGUFType type);

std::any ReadGGUFArray(::ohllm::common::BinaryFileReader &freader);

struct KVPair {
  std::string name;
  GGUFType type;
  std::any value;
};

using KVPairs = std::vector<KVPair>;

void KVPairs2ModelConfig(const KVPairs &kv_pairs,
                         ModelConfig *config = nullptr);

struct TensorInfo {
  std::string name;
  std::vector<U64> shape;
  GGMLType type;
  U64 offset;
  U64 nbytes;
};
using TensorInfos = std::vector<TensorInfo>;

}  // namespace internal


void GGUFModelLoader::Load(const std::string &filename,
                           std::vector<char> *buffer, StateDict *state_dict,
                           ModelConfig *config) {
  // auto fin = std::ifstream(filename, std::ios::binary);
  auto freader = ::ohllm::common::BinaryFileReader(filename);
  ASSERT(freader.good(), "Failed to open (%s)", filename.c_str());

  {  // check magic
    std::string magic = freader.ReadString(4);
    ASSERT(GGUF_MAGIC == magic, "Invalid gguf magic (%s)", magic.c_str());
  }


  U32 version = freader.Read<U32>();
  ASSERT(version > 1, "Unsupported gguf version (%d)", version);
  U64 n_tensors = freader.Read<U64>();
  U64 n_kv = freader.Read<U64>();
  ASSERT(freader.good(), "Failed to read gguf header");

  U64 alignment = GGUF_DEFAULT_ALIGNMENT;
  internal::KVPairs kv_pairs;
  {  // read kv pairs
    for (U64 i = 0; i < n_kv; ++i) {
      auto key = freader.ReadString(freader.Read<U64>());
      auto type = freader.Read<internal::GGUFType>();
      std::any value = internal::ReadGGUF(freader, type);
      if (key == "general.alignment") {
        ASSERT(type == internal::GGUF_TYPE_UINT32, "");
        alignment = std::any_cast<U32>(value);
      }
      kv_pairs.push_back({key, type, value});
    }
    ASSERT(freader.good(), "Failed to read gguf key-value pairs");
  }
  internal::KVPairs2ModelConfig(kv_pairs, config);

  internal::TensorInfos tensor_infos;
  U64 ctx_size = 0;
  {  // read tensor infos
    for (U64 i = 0; i < n_tensors; ++i) {
      auto name = freader.ReadString(freader.Read<U64>());
      auto ndim = freader.Read<U32>();
      auto shape = freader.ReadVector<U64>(ndim);
      std::reverse(shape.begin(), shape.end());
      auto type = freader.Read<internal::GGMLType>();
      auto offset = freader.Read<U64>();
      ASSERT(offset % GGUF_DEFAULT_ALIGNMENT == 0, "");
      U64 ne = std::accumulate(shape.begin(), shape.end(), 1,
                               std::multiplies<U64>());
      auto type_info = internal::kGGMLTypeInfos.at(type);
      U64 size_cur = type_info.nbytes * ne / type_info.block_size;
      ctx_size += GGML_PAD(size_cur, alignment);
      tensor_infos.push_back({name, shape, type, offset, size_cur});
    }
    ASSERT(freader.good(), "Failed to read gguf tensor infos");
  }

  {  // make alignment
    U64 pos = freader.tellg();
    freader.seekg((pos - 1) / alignment * alignment + alignment);
    ASSERT(freader.good(), "Failed to make gguf alignment");
  }

  {  // read tensors
    buffer->resize(ctx_size);
    auto data = buffer->data();
    freader.ReadBytes(data, ctx_size);
    ASSERT(freader.good(), "Failed to read gguf ctx");
    for (U64 i = 0; i < n_tensors; ++i) {
      const auto &tensor_info = tensor_infos[i];
      std::vector<I32> shape;
      for (auto s : tensor_info.shape) { shape.push_back(static_cast<I32>(s)); }
      const auto &type_info = internal::kGGMLTypeInfos.at(tensor_info.type);
      core::Tensor tensor(shape, data + tensor_info.offset, type_info.dtype,
                          tensor_info.name);
      ASSERT(tensor_info.nbytes == tensor.nbytes(), "");
      state_dict->AddTensor(tensor_info.name, tensor);
    }
  }
}

namespace internal {

std::any ReadGGUF(::ohllm::common::BinaryFileReader &freader, GGUFType type) {
  if (type != GGUF_TYPE_ARRAY) {
    switch (type) {
      case GGUF_TYPE_UINT8:
        return freader.Read<U8>();
      case GGUF_TYPE_INT8:
        return freader.Read<I8>();
      case GGUF_TYPE_UINT16:
        return freader.Read<U16>();
      case GGUF_TYPE_INT16:
        return freader.Read<I16>();
      case GGUF_TYPE_UINT32:
        return freader.Read<U32>();
      case GGUF_TYPE_INT32:
        return freader.Read<I32>();
      case GGUF_TYPE_FLOAT32:
        return freader.Read<F32>();
      case GGUF_TYPE_BOOL:
        return freader.Read<bool>();
      case GGUF_TYPE_UINT64:
        return freader.Read<U64>();
      case GGUF_TYPE_INT64:
        return freader.Read<I64>();
      case GGUF_TYPE_FLOAT64:
        return freader.Read<F64>();
      case GGUF_TYPE_STRING:
        return freader.ReadString(freader.Read<U64>());
      default:
        ASSERT(false, "");
    }
  }
  return ReadGGUFArray(freader);
}


std::any ReadGGUFArray(::ohllm::common::BinaryFileReader &freader) {
  auto type = freader.Read<GGUFType>();
  ASSERT(type != GGUF_TYPE_ARRAY && type != GGUF_TYPE_COUNT, "");
  auto n = freader.Read<U64>();
  if (type != GGUF_TYPE_STRING) {
    switch (type) {
      case GGUF_TYPE_UINT8:
        return freader.ReadVector<U8>(n);
      case GGUF_TYPE_INT8:
        return freader.ReadVector<I8>(n);
      case GGUF_TYPE_UINT16:
        return freader.ReadVector<U16>(n);
      case GGUF_TYPE_INT16:
        return freader.ReadVector<I16>(n);
      case GGUF_TYPE_UINT32:
        return freader.ReadVector<U32>(n);
      case GGUF_TYPE_INT32:
        return freader.ReadVector<I32>(n);
      case GGUF_TYPE_FLOAT32:
        return freader.ReadVector<F32>(n);
      case GGUF_TYPE_BOOL:
        return freader.ReadVector<bool>(n);
      case GGUF_TYPE_UINT64:
        return freader.ReadVector<U64>(n);
      case GGUF_TYPE_INT64:
        return freader.ReadVector<I64>(n);
      case GGUF_TYPE_FLOAT64:
        return freader.ReadVector<F64>(n);
      default:
        ASSERT(false, "");
    }
  }
  std::vector<std::string> buf(n);
  for (U64 i = 0; i < n; ++i) {
    buf[i] = freader.ReadString(freader.Read<U64>());
  }
  return buf;
}

void KVPairs2ModelConfig(const KVPairs &kv_pairs, ModelConfig *config) {
  if (config == nullptr) { return; }
  for (const auto &kv : kv_pairs) {
    const auto &name = kv.name;
    const auto &type = kv.type;
    const auto &value = kv.value;
    LOGI("%s: %s", name.c_str(), kGGUFTypeNames[type].c_str());
    // basic
    if (name == "general.architecture") {
      ASSERT(type == GGUF_TYPE_STRING, "");
      config->model_arch_type = std::any_cast<std::string>(value);
      continue;
    }
    if (name == "general.name") {
      ASSERT(type == GGUF_TYPE_STRING, "");
      config->model_name = std::any_cast<std::string>(value);
      continue;
    }
    if (name.ends_with("block_count")) {
      ASSERT(type == GGUF_TYPE_UINT32, "");
      config->num_blocks = std::any_cast<U32>(value);
      continue;
    }
    if (name.ends_with("embedding_length")) {
      ASSERT(type == GGUF_TYPE_UINT32, "");
      config->hidden_size = std::any_cast<U32>(value);
      continue;
    }
    // pe
    if (name.ends_with("context_length")) {
      ASSERT(type == GGUF_TYPE_UINT32, "");
      config->pe.context_length = std::any_cast<U32>(value);
      continue;
    }
    if (name.ends_with("rope.freq_base")) {
      ASSERT(type == GGUF_TYPE_FLOAT32, "");
      config->pe.rope_base = std::any_cast<F32>(value);
      continue;
    }
    if (name.ends_with("rope.dimension_count")) {
      ASSERT(type == GGUF_TYPE_UINT32, "");
      config->pe.rope_dim = std::any_cast<U32>(value);
      continue;
    }
    // attention
    if (name.ends_with("attention.head_count")) {
      ASSERT(type == GGUF_TYPE_UINT32, "");
      config->attn.num_heads = std::any_cast<U32>(value);
      continue;
    }
    if (name.ends_with("attention.head_count_kv")) {
      ASSERT(type == GGUF_TYPE_UINT32, "");
      config->attn.num_kv_heads = std::any_cast<U32>(value);
      continue;
    }
    // ffn
    if (name.ends_with("feed_forward_length")) {
      ASSERT(type == GGUF_TYPE_UINT32, "");
      config->ffn.intermediate_size = std::any_cast<U32>(value);
      continue;
    }
    // layer norm
    if (name.ends_with("attention.layer_norm_rms_epsilon")) {
      ASSERT(type == GGUF_TYPE_FLOAT32, "");
      config->ln.eps = std::any_cast<F32>(value);
      continue;
    }
    // tokenizer
    if (name.ends_with("bos_token_id")) {
      ASSERT(type == GGUF_TYPE_UINT32, "");
      config->tokenlizer.bos_token_id = std::any_cast<U32>(value);
      continue;
    }
    if (name.ends_with("eos_token_id")) {
      ASSERT(type == GGUF_TYPE_UINT32, "");
      config->tokenlizer.eos_token_id = std::any_cast<U32>(value);
      continue;
    }
    if (name.ends_with("padding_token_id")) {
      ASSERT(type == GGUF_TYPE_UINT32, "");
      config->tokenlizer.pad_token_id = std::any_cast<U32>(value);
      continue;
    }
    if (name == "tokenizer.chat_template") {
      ASSERT(type == GGUF_TYPE_STRING, "");
      config->tokenlizer.chat_template = std::any_cast<std::string>(value);
      continue;
    }
  }
}

}  // namespace internal

}  // namespace ohllm::model
