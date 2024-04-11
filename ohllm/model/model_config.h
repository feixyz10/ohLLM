#pragma once

#include <string>

#include "ohllm/common/type.h"

namespace ohllm::model {

namespace internal {

struct PositionEmbConfig {
  I32 context_length;
  I32 rope_base;
  F32 rope_dim = 0;  // default if <= 0: hidden_size // num_heads
};

struct AttentionConfig {
  I32 num_heads;
  I32 num_kv_heads = 0;  // default if <=0:
};

struct FFNConfig {
  I32 intermediate_size;
  std::string activation_type;  // [silu, gelu, ...]
};

struct LayerNormConfig {
  F32 eps;
  std::string type;  // [layer_norm, rms_norm]
};

struct TokenlizerConfig {
  I32 bos_token_id;
  I32 eos_token_id;
  I32 pad_token_id;
  I32 num_tokens;
  std::string chat_template;
};

}  // namespace internal

struct ModelConfig {
  // basic
  std::string model_arch_type;  // [llama, qwen, phi, ...]
  std::string model_name;
  I32 num_blocks;
  I32 hidden_size;
  I32 vocab_size;
  bool tie_word_embeddings = false;

  internal::PositionEmbConfig pe;

  internal::AttentionConfig attn;

  internal::FFNConfig ffn;

  internal::LayerNormConfig ln;

  internal::TokenlizerConfig tokenlizer;
};

}  // namespace ohllm::model