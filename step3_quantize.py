"""
Step 3: 量化 — AWQ INT4 (vision encoder 用 INT8)
================================================
針對 ASIC/CIM 部署做量化，vision 和 language 分開處理。

Usage:
    python step3_quantize.py --model pruned_model/ --output quantized_model/
    python step3_quantize.py --model Qwen/Qwen2-VL-7B-Instruct \
                             --output quantized_model/   # 也可以直接量化原始模型
"""

import argparse
import json
import sys
from pathlib import Path


def check_dependencies():
    """檢查必要的套件是否安裝。"""
    missing = []
    try:
        import torch
    except ImportError:
        missing.append("torch")
    try:
        import transformers
    except ImportError:
        missing.append("transformers")

    # AWQ is optional — check and report
    has_awq = False
    try:
        import awq
        has_awq = True
    except ImportError:
        pass

    if missing:
        print(f"Missing packages: {missing}")
        print(f"Run: pip install {' '.join(missing)}")
        sys.exit(1)

    return has_awq


def quantize_with_awq(model_path, output_path, w_bit=4, group_size=128):
    """使用 AWQ 做 INT4 量化。"""
    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer

    print(f"Loading model for AWQ quantization: {model_path}")
    model = AutoAWQForCausalLM.from_pretrained(model_path, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    quant_config = {
        "zero_point": True,
        "q_group_size": group_size,
        "w_bit": w_bit,
        "version": "GEMM",
    }

    print(f"Quantizing to INT{w_bit} (group_size={group_size})...")
    model.quantize(tokenizer, quant_config=quant_config)

    print(f"Saving to: {output_path}")
    model.save_quantized(output_path)
    tokenizer.save_pretrained(output_path)

    return output_path


def quantize_manual(model_path, output_path, lm_bits=4, vision_bits=8):
    """
    手動量化：直接對 state_dict 做 absmax/zero-point 量化。
    不需要 GPU，在 CPU/MPS 上就能跑。
    """
    import torch
    from pathlib import Path

    print(f"Loading model: {model_path}")
    print(f"  Language model: INT{lm_bits}")
    print(f"  Vision encoder: INT{vision_bits}")

    model_path = Path(model_path)
    state_dict_path = model_path / "pytorch_model.bin"
    if not state_dict_path.exists():
        # Try safetensors or download
        from transformers import Qwen2VLForConditionalGeneration
        print("  Loading from HuggingFace...")
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            str(model_path), torch_dtype=torch.float32, device_map=None
        ).cpu()
        state_dict = model.state_dict()
        del model
    else:
        print(f"  Loading state dict from: {state_dict_path}")
        state_dict = torch.load(state_dict_path, map_location="cpu", weights_only=True)

    total_original_bytes = sum(v.nbytes for v in state_dict.values())
    total_params = sum(v.numel() for v in state_dict.values())

    print(f"  Total params: {total_params / 1e9:.2f}B")
    print(f"  Original size: {total_original_bytes / (1024**3):.2f} GB")

    # Quantize each weight tensor
    quantized_dict = {}
    scale_dict = {}
    zero_dict = {}
    stats = {"lm_quantized": 0, "vis_quantized": 0, "kept_fp": 0}

    for name, tensor in state_dict.items():
        # Skip non-weight tensors (biases, layernorm, embeddings)
        is_weight = ("weight" in name and tensor.dim() >= 2
                     and "norm" not in name and "embed" not in name)

        if not is_weight:
            quantized_dict[name] = tensor.to(torch.float16)
            stats["kept_fp"] += 1
            continue

        # Determine bit width
        is_vision = "visual" in name
        bits = vision_bits if is_vision else lm_bits

        if bits == 4:
            qmin, qmax = -8, 7
        else:  # INT8
            qmin, qmax = -128, 127

        # Per-channel absmax quantization
        flat = tensor.float()
        scale = flat.abs().amax(dim=-1, keepdim=True) / (-qmin)
        scale = scale.clamp(min=1e-10)
        quantized = (flat / scale).round().clamp(qmin, qmax).to(torch.int8)

        quantized_dict[name] = quantized
        scale_dict[name] = scale.to(torch.float16).squeeze(-1)

        if is_vision:
            stats["vis_quantized"] += 1
        else:
            stats["lm_quantized"] += 1

    # Calculate compressed size
    total_quantized_bytes = 0
    for name, tensor in quantized_dict.items():
        total_quantized_bytes += tensor.nbytes
    for name, tensor in scale_dict.items():
        total_quantized_bytes += tensor.nbytes

    print(f"\nQuantization results:")
    print(f"  LM layers quantized:     {stats['lm_quantized']}")
    print(f"  Vision layers quantized:  {stats['vis_quantized']}")
    print(f"  Kept in FP16:            {stats['kept_fp']}")
    print(f"  Original size:  {total_original_bytes / (1024**3):.2f} GB")
    print(f"  Quantized size: {total_quantized_bytes / (1024**3):.2f} GB")
    print(f"  Compression:    {total_original_bytes / total_quantized_bytes:.1f}x")

    # Save
    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=True)

    torch.save({
        "quantized_state_dict": quantized_dict,
        "scales": scale_dict,
    }, output / "quantized_model.bin")

    # Copy config and processor files
    import shutil
    for fname in ["config.json", "tokenizer.json", "tokenizer_config.json",
                  "processor_config.json", "chat_template.jinja"]:
        src = model_path / fname
        if src.exists():
            shutil.copy2(src, output / fname)

    # Save metadata
    meta = {
        "original_model": str(model_path),
        "language_model_bits": lm_bits,
        "vision_encoder_bits": vision_bits,
        "total_params_B": total_params / 1e9,
        "original_size_GB": total_original_bytes / (1024 ** 3),
        "quantized_size_GB": total_quantized_bytes / (1024 ** 3),
        "compression_ratio": total_original_bytes / total_quantized_bytes,
        "method": "absmax_per_channel",
        "lm_layers_quantized": stats["lm_quantized"],
        "vision_layers_quantized": stats["vis_quantized"],
        "kept_fp16": stats["kept_fp"],
    }
    with open(output / "quant_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved to: {output}")
    return output


def estimate_compression(model_path):
    """估算量化後的壓縮比，包含 CIM array 對應。"""
    print("\n" + "=" * 70)
    print("COMPRESSION ESTIMATE FOR ASIC/CIM")
    print("=" * 70)

    # Qwen2-VL-7B: ~8.9B params in BF16
    original_size_gb = 8.9e9 * 2 / (1024 ** 3)  # BF16 = 2 bytes/param

    # Qwen2-VL-7B key dimensions
    # LM: hidden=3584, intermediate=18944, heads=28, kv_heads=4, layers=28
    # Vision: embed_dim=1280, mlp_ratio=4, heads=16, layers=32
    lm_hidden = 3584
    lm_inter = 18944
    lm_layers = 28
    vis_hidden = 1280

    # scenarios: (name, bits, keep_ratio, compression, lm_layers_kept)
    scenarios = [
        ("Original BF16",           16, 1.0,  1.0,  28),
        ("INT8 only",                8, 1.0,  2.0,  28),
        ("INT4 only",                4, 1.0,  4.0,  28),
        ("Prune 20% + INT8",         8, 0.8,  2.5,  22),
        ("Prune 20% + INT4",         4, 0.8,  5.0,  22),
        ("Prune 20% + 2:4 + INT4",   4, 0.8,  10.0, 22),
        ("Prune 25% + INT4 (CIM)",   4, 0.75, 5.3,  21),
    ]

    print(f"\n  {'Scenario':<28} {'Bits':>5} {'Size':>7} {'Comp.':>6} {'LM Layers':>10}")
    print("  " + "-" * 60)
    for name, bits, keep_ratio, compression, layers_kept in scenarios:
        size = original_size_gb * (bits / 16) * keep_ratio
        print(f"  {name:<28} {bits:>3}bit {size:>5.1f}GB {compression:>5.1f}x {layers_kept:>10}")

    # CIM array mapping
    print(f"\n" + "=" * 70)
    print("CIM ARRAY SIZE MAPPING — Qwen2-VL-7B")
    print("=" * 70)
    print("""
  Qwen2-VL-7B 的主要矩陣運算與 CIM 陣列對應：

  ┌─────────────────────────────────────────────────────────────────┐
  │ Component          Matrix Shape       CIM Array    Tiles Needed │
  ├─────────────────────────────────────────────────────────────────┤
  │ LM: QKV Projection (GQA 28:4)                                  │
  │   Q_proj            3584 × 3584       256×256         196      │
  │   K_proj (GQA)      3584 ×  512       256×256          28      │
  │   V_proj (GQA)      3584 ×  512       256×256          28      │
  │   O_proj            3584 × 3584       256×256         196      │
  │                                                                 │
  │ LM: FFN (SwiGLU)                                               │
  │   gate_proj         3584 × 18944      256×256        1036      │
  │   up_proj           3584 × 18944      256×256        1036      │
  │   down_proj        18944 × 3584       256×256        1036      │
  │                                                                 │
  │ Vision: ViT                                                     │
  │   QKV_proj          1280 × 1280       256×256          25      │
  │   FFN_up            1280 × 5120       256×256         100      │
  │   FFN_down          5120 × 1280       256×256         100      │
  │                                                                 │
  │ PatchMerger                                                     │
  │   projection        5120 × 3584       256×256         280      │
  │   (無 cross-attention — Qwen2-VL 用 M-RoPE fusion)             │
  └─────────────────────────────────────────────────────────────────┘

  * Tiles = ceil(M/256) × ceil(N/256)，假設 CIM array 為 256×256
  * GQA 28:4 = K/V 只有 4 heads (vs Q 的 28 heads)，矩陣更小
  * 無 cross-attention layers → 比 LLaMA Vision 少很多運算""")

    # Per-layer MAC operations for Qwen2-VL-7B
    # Q: 3584*3584=12.8M, K: 3584*512=1.8M, V: 3584*512=1.8M, O: 3584*3584=12.8M = 29.3M
    # FFN: 3*3584*18944 = 203.7M
    # ViT per layer: 4*1280*1280 + 2*1280*5120 = 19.7M
    print(f"""
  ┌─────────────────────────────────────────────────────────────────┐
  │                Per-Layer MAC Operations (Qwen2-VL-7B)           │
  ├─────────────────────────────────────────────────────────────────┤
  │ Component              MACs/token    ×Layers    Total MACs      │
  ├─────────────────────────────────────────────────────────────────┤
  │ LM Self-Attn proj     29.3M          ×28         820M          │
  │ LM FFN (SwiGLU)      203.7M          ×28       5,704M          │
  │ Vision ViT             19.7M          ×32         630M          │
  │ PatchMerger            18.4M          ×1           18M          │
  ├─────────────────────────────────────────────────────────────────┤
  │ Total per token                                 7,172M MACs     │
  └─────────────────────────────────────────────────────────────────┘

  剪枝後 (20% LM layers removed, 28→22):
  │ LM Self-Attn proj     29.3M          ×22         644M  (-21%)  │
  │ LM FFN (SwiGLU)      203.7M          ×22       4,481M  (-21%)  │
  │ Total per token                                 5,773M MACs     │
  │ → 省下 ~20% 計算量，無 cross-attention 開銷更低                 │""")

    # CIM array size recommendations
    print(f"""
  ┌─────────────────────────────────────────────────────────────────┐
  │ CIM Array Size Recommendations                                  │
  ├─────────────────────────────────────────────────────────────────┤
  │ Array Size   Bit Width   SRAM/cell   適合場景                    │
  ├─────────────────────────────────────────────────────────────────┤
  │ 256 × 256    INT4        4-bit       最佳平衡：LM FFN 整除       │
  │ 256 × 256    INT8        8-bit       Vision encoder 精度需求     │
  │ 512 × 512    INT4        4-bit       大陣列：減少 tiling 開銷     │
  │ 128 × 128    INT4        4-bit       面積受限：需更多 tiling      │
  └─────────────────────────────────────────────────────────────────┘

  建議：
  → 256×256 INT4 陣列是最佳選擇
     - LM hidden_size=3584 = 14 tiles per row（整除）
     - 剪枝後 22 layers × 每層 ~3,556 tiles = ~78K tiles total
  → Vision encoder 用獨立的 256×256 INT8 陣列
     - embed_dim=1280 = 5 tiles per row
  → 3584 = 256 × 14（完美整除），比 LLaMA 的 4096 更省 tiles
  → 無 cross-attention = 省下大量 CIM tiles 和 data movement
    """)


def main():
    parser = argparse.ArgumentParser(description="Quantize model for ASIC/CIM deployment")
    parser.add_argument("--model", default="Qwen/Qwen2-VL-7B-Instruct",
                        help="Model path (pruned or original)")
    parser.add_argument("--output", default="quantized_model/")
    parser.add_argument("--lm-bits", type=int, default=4, choices=[4, 8],
                        help="Language model quantization bits (default: 4)")
    parser.add_argument("--vision-bits", type=int, default=8, choices=[4, 8],
                        help="Vision encoder quantization bits (default: 8)")
    parser.add_argument("--method", default="auto", choices=["auto", "awq", "bnb"],
                        help="Quantization method")
    parser.add_argument("--estimate-only", action="store_true",
                        help="Only show compression estimates")
    args = parser.parse_args()

    if args.estimate_only:
        estimate_compression(args.model)
        return

    has_awq = check_dependencies()

    import platform
    is_mac = platform.system() == "Darwin"

    if args.method == "auto":
        if is_mac:
            args.method = "bnb"  # manual absmax on Mac (no CUDA)
        elif has_awq:
            args.method = "awq"
        else:
            args.method = "bnb"

    if args.method == "awq" and not has_awq:
        print("AWQ not installed. Run: pip install autoawq")
        sys.exit(1)

    print(f"Using quantization method: {args.method}")
    if is_mac:
        print("  (Mac detected — using CPU absmax quantization)")

    if args.method == "awq":
        quantize_with_awq(args.model, args.output, w_bit=args.lm_bits)
    else:
        quantize_manual(args.model, args.output,
                        lm_bits=args.lm_bits, vision_bits=args.vision_bits)

    estimate_compression(args.model)
    print("\nDone! Next step: export to ONNX for ASIC compiler.")


if __name__ == "__main__":
    main()
