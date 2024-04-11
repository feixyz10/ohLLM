import gguf
import torch
import safetensors.torch
import numpy as np
from pathlib import Path

# def quantize_q40(w: np.ndarray, block_size=32, block_nbytes=18):
#     numel = w.size
#     assert numel % block_size == 0
#     w = w.reshape(-1, block_size)
#     amax = np.max(np.abs(w), axis=1)
#     d = amax / 15
#     invd = np.copy(d)
#     invd[d != 0] = 1 / invd[d != 0]
#     wq = w * invd.reshape(-1, 1)
#     wq = np.clip((wq + 8.5).astype(np.int8), 0, 15).astype(np.uint8)
#     bytesarr = b""
#     for i in range(numel // block_nbytes):
#         bytesarr += np.float16(d[i]).tobytes()
#         bytesarr += wq[i].tobytes()
#     return np.frombuffer(bytesarr, dtype=np.uint8)


def quantize_q80(w: np.ndarray, block_size=32, block_nbytes=34):
    numel = w.size
    shape = w.shape
    assert numel % block_size == 0
    w = w.reshape(-1, block_size)
    amax = np.max(np.abs(w), axis=1)
    d = amax / 127
    invd = np.copy(d)
    invd[d != 0] = 1 / invd[d != 0]
    wq = w * invd.reshape(-1, 1)
    wq = np.round(wq).astype(np.int8)
    ret = np.zeros((w.shape[0], block_nbytes), dtype=np.uint8)
    for i in range(w.shape[0]):
        bytesarr = b""
        bytesarr += np.float16(d[i]).tobytes()
        bytesarr += wq[i].tobytes()
        ret[i] = np.frombuffer(bytesarr, dtype=np.uint8)
    return ret


def quantize(tensor: torch.Tensor, quant_type="Q8_0"):
    assert quant_type in ["FP32", "FP16", "Q8_0", "Q4_0", "Q4_1"]
    assert tensor.is_contiguous()
    tensor: np.ndarray = tensor.flatten().float().numpy()
    if quant_type == "FP32":
        return tensor.astype(np.float32)
    if quant_type == "FP16":
        return tensor.astype(np.float16)
    if quant_type == "Q8_0":
        return quantize_q80(tensor)
    else:
        raise NotImplementedError


QTYPE_MAP = {
    "FP32": gguf.GGMLQuantizationType.F32,
    "FP16": gguf.GGMLQuantizationType.F16,
    "Q8_0": gguf.GGMLQuantizationType.Q8_0,
    "Q4_0": gguf.GGMLQuantizationType.Q4_0,
    "Q4_1": gguf.GGMLQuantizationType.Q4_1,
}


def convert(in_path: Path, out_path: Path, quant_type="Q8_0", arch: str = ""):
    state_dicts = safetensors.torch.load_file(in_path)
    gguf_writer = gguf.GGUFWriter(out_path, arch)
    for i, (name, tensor) in enumerate(state_dicts.items()):
        tensor = tensor.float()
        print(f"processing {i+1}/{len(state_dicts)}: {name}")
        shape = tensor.shape
        tensor = quantize(tensor, quant_type)
        gguf_writer.add_tensor(
            name, tensor, raw_shape=shape, raw_dtype=QTYPE_MAP[quant_type]
        )
    gguf_writer.write_header_to_file()
    gguf_writer.write_kv_data_to_file()
    gguf_writer.write_tensors_to_file()
    gguf_writer.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_in", type=str, default="")
    parser.add_argument("--model_out", type=str, default="")
    parser.add_argument(
        "--quant_type", type=str, choices=["FP32", "FP16", "Q8_0", "Q4_0", "Q4_1"]
    )
    args = parser.parse_args()
    print(args)

    path_proj = Path(__file__).parent.parent
    path_model = path_proj / ".temp"
    model_fn = path_model / "Qwen1.5-0.5B-Chat/model.safetensors"

    convert(
        model_fn,
        path_model / (model_fn.parent.name + "_" + args.quant_type.lower() + ".gguf"),
        quant_type=args.quant_type,
        arch=model_fn.parent.name,
    )
