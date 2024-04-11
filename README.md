# OhLLM

Pure C++ LLM inference engine. **No third party dependencies.**

* under construction, for education and learning purpose only (currently). Hope it will be eventually evolved to be useful in production.*

License: MIT

## Build

```bash
./scripts/build.sh [--debug] [--test] [--nolog] [--clear]

# argments:
#   --debug: build debug version for debugging
#   --test: run all unit tests
#   --nolog: build without log
#   --clear: clear and rebuild
```

## Usage

## Benchmark

## Features

This project will mainly focus on LLM inference on single machine with/without accelerator (Nvidia GPU, apple MPS, ...).

As the project proceeds:

- support cpu inference in FP32/FP16, no vectorization intrinsics, compatible with GGUF format
- support mainstream LLM models, like llama, phi, qwen, gemma and ...
- support cpu inference in FP32/FP16 plus vectorization intrinsics
- support cpu inference with quantization of Q8_0, Q4_0 and Q4_1
- support streaming output
- support nvidia gpu inference
- support nvidia gpu inference with cpu offloading
- support apple mps

## Acknowledgements

This project is greatly inspired by [**ggml**](https://github.com/ggerganov/ggml), [**fastllm**](https://github.com/ztxz16/fastllm) and [**llama.cpp**](https://github.com/ggerganov/llama.cpp). Thanks a lot to all of the contributors.

Please buy me a [:coffee:](https://ko-fi.com/excitingme) if you find this project useful. 