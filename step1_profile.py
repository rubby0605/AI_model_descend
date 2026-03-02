"""
Step 1: Profile Qwen2-VL-7B 模型架構
=====================================
分析各 component 的參數量、記憶體佔用，為後續剪枝提供 baseline。

Qwen2-VL 架構：ViT encoder + PatchMerger + Qwen2 LM（無 cross-attention，用 M-RoPE fusion）

Usage:
    python step1_profile.py [--model Qwen/Qwen2-VL-7B-Instruct]
    python step1_profile.py --dry-run   # 不下載模型，只看 config
"""

import argparse
import json
import sys
from collections import defaultdict


def profile_config_only(model_name):
    """從 config 估算參數量，不需要下載模型權重。"""
    from transformers import AutoConfig

    print(f"[Config-only mode] Loading config: {model_name}")
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

    print("\n" + "=" * 60)
    print("MODEL CONFIGURATION — Qwen2-VL")
    print("=" * 60)

    # Vision encoder config
    if hasattr(config, 'vision_config'):
        vc = config.vision_config
        depth = getattr(vc, 'depth', None)
        embed_dim = getattr(vc, 'embed_dim', None)
        num_heads = getattr(vc, 'num_heads', None)
        mlp_ratio = getattr(vc, 'mlp_ratio', 4)
        patch_size = getattr(vc, 'patch_size', getattr(vc, 'spatial_patch_size', 14))
        merge_size = getattr(vc, 'spatial_merge_size', 2)

        print("\n--- Vision Encoder (ViT) ---")
        print(f"  Embed dim:         {embed_dim}")
        print(f"  MLP ratio:         {mlp_ratio}")
        print(f"  Num layers:        {depth}")
        print(f"  Attention heads:   {num_heads}")
        print(f"  Patch size:        {patch_size}")
        print(f"  Spatial merge:     {merge_size}×{merge_size}")
        print(f"  Window attn:       layers except [7, 15, 23, 31] use windowed attention")

        if embed_dim and depth:
            ff = int(embed_dim * mlp_ratio)
            # Per layer: QKV + out proj + FFN (SwiGLU: gate + up + down) + norms
            per_layer = 4 * embed_dim * embed_dim + 3 * embed_dim * ff + 4 * embed_dim
            vision_est = depth * per_layer
            # PatchMerger: project from embed_dim*merge^2 to hidden_size
            lm_hidden = getattr(config, 'hidden_size', 3584)
            merger_est = embed_dim * (merge_size ** 2) * lm_hidden
            print(f"  Estimated ViT params:     ~{vision_est / 1e6:.0f}M")
            print(f"  Estimated Merger params:  ~{merger_est / 1e6:.0f}M")

    # Language model config (Qwen2-VL: LM params are at top level, not in text_config)
    lm_hidden = getattr(config, 'hidden_size', None)
    lm_inter = getattr(config, 'intermediate_size', None)
    lm_layers = getattr(config, 'num_hidden_layers', None)
    lm_heads = getattr(config, 'num_attention_heads', None)
    lm_kv_heads = getattr(config, 'num_key_value_heads', None)
    lm_vocab = getattr(config, 'vocab_size', None)
    lm_max_pos = getattr(config, 'max_position_embeddings', None)

    if lm_hidden:
        print("\n--- Language Model (Qwen2) ---")
        print(f"  Hidden size:       {lm_hidden}")
        print(f"  Intermediate size: {lm_inter}")
        print(f"  Num layers:        {lm_layers}")
        print(f"  Attention heads:   {lm_heads}")
        print(f"  KV heads:          {lm_kv_heads} (GQA ratio {lm_heads}:{lm_kv_heads})")
        print(f"  Vocab size:        {lm_vocab}")
        print(f"  Max positions:     {lm_max_pos}")

        head_dim = lm_hidden // lm_heads
        qkvo = lm_hidden * (lm_heads * head_dim) + lm_hidden * (lm_kv_heads * head_dim) * 2 + lm_hidden * lm_hidden
        ffn = 3 * lm_hidden * lm_inter  # gate + up + down (SwiGLU)
        per_layer = qkvo + ffn + 2 * lm_hidden
        lm_est = lm_layers * per_layer + lm_vocab * lm_hidden
        print(f"  Estimated params:  ~{lm_est / 1e6:.0f}M")

    # M-RoPE config
    rope = getattr(config, 'rope_scaling', None)
    if rope:
        print(f"\n--- M-RoPE (Multimodal Rotary Position Embedding) ---")
        print(f"  Type:              {rope.get('type', 'N/A')}")
        print(f"  mrope_section:     {rope.get('mrope_section', 'N/A')}")
        print(f"  → 無 cross-attention，用 M-RoPE 做 vision-language fusion")

    print(f"\n--- Fusion 方式 ---")
    print(f"  Qwen2-VL 不用 cross-attention（跟 LLaMA Vision 不同）")
    print(f"  Vision tokens 經 PatchMerger 壓縮後直接跟 text tokens 拼接")
    print(f"  M-RoPE 分 3 軸: [temporal, height, width] 編碼位置資訊")
    print(f"  → 剪枝不需要保護 cross-attention 層，比 LLaMA 簡單")

    print("\n" + "=" * 60)
    print("Full config JSON (truncated):")
    print(json.dumps(json.loads(config.to_json_string()), indent=2, ensure_ascii=False)[:3000])
    print("... (truncated)")


def profile_full_model(model_name):
    """載入完整模型，精確計算每個 component 的參數量。"""
    import torch
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

    print(f"[Full mode] Loading model: {model_name}")
    print("This may take a while and requires significant GPU/RAM...\n")

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    # Categorize parameters
    components = defaultdict(lambda: {"params": 0, "size_mb": 0, "layers": set()})

    for name, param in model.named_parameters():
        n_params = param.numel()
        size_mb = param.nbytes / (1024 * 1024)

        if "visual" in name:
            if "merger" in name:
                cat = "patch_merger"
            else:
                cat = "vision_encoder"
        elif "embed_tokens" in name or "lm_head" in name:
            cat = "embedding_head"
        else:
            cat = "language_model"

        components[cat]["params"] += n_params
        components[cat]["size_mb"] += size_mb

        # Track layer indices
        parts = name.split(".")
        for i, p in enumerate(parts):
            if p in ("layers", "blocks") and i + 1 < len(parts) and parts[i + 1].isdigit():
                components[cat]["layers"].add(int(parts[i + 1]))

    # Print results
    total_params = sum(c["params"] for c in components.values())
    total_mb = sum(c["size_mb"] for c in components.values())

    print("=" * 70)
    print("MODEL PROFILE — Qwen2-VL-7B")
    print("=" * 70)
    print(f"{'Component':<25} {'Params':>12} {'Size (MB)':>12} {'% Total':>10} {'Layers':>10}")
    print("-" * 70)

    for cat in ["vision_encoder", "patch_merger", "language_model", "embedding_head"]:
        if cat not in components:
            continue
        c = components[cat]
        pct = c["params"] / total_params * 100
        n_layers = len(c["layers"]) if c["layers"] else "-"
        print(f"{cat:<25} {c['params']:>12,} {c['size_mb']:>10.1f}MB {pct:>9.1f}% {str(n_layers):>10}")

    print("-" * 70)
    print(f"{'TOTAL':<25} {total_params:>12,} {total_mb:>10.1f}MB {100.0:>9.1f}%")
    print("=" * 70)

    # Pruning recommendations
    print("\n" + "=" * 70)
    print("PRUNING RECOMMENDATIONS FOR ASIC/CIM")
    print("=" * 70)

    lm = components["language_model"]
    ve = components["vision_encoder"]
    pm = components.get("patch_merger", {"params": 0})

    print(f"""
    1. Language Model ({lm['params']/1e9:.1f}B, {lm['params']/total_params*100:.0f}% of total)
       → 主要剪枝目標。ShortGPT 可砍 20-25% layers (28→22 layers)。
       → 無 cross-attention 依賴，任何 layer 都可以被移除。
       → 結構化剪枝 + INT4 量化效果最好。

    2. Vision Encoder ({ve['params']/1e9:.1f}B, {ve['params']/total_params*100:.0f}% of total)
       → 佔比較小，不建議剪太多。
       → 保留 full attention layers [7, 15, 23, 31]，windowed layers 可剪。
       → 量化用 INT8（比 language model 對量化敏感）。

    3. PatchMerger ({pm['params']/1e6:.0f}M, 極小)
       → 不需剪枝，保持完整。

    Target compression for ASIC/CIM:
       → 結構化剪枝 20% + INT4 量化 = ~5x compression
       → 預期保留 ~90% 原始精度
       → 無 cross-attention = 剪枝比 LLaMA Vision 簡單
    """)

    # Save detailed breakdown
    detail_path = "profile_detail.json"
    detail = {}
    for name, param in model.named_parameters():
        detail[name] = {
            "shape": list(param.shape),
            "params": param.numel(),
            "dtype": str(param.dtype),
        }
    with open(detail_path, "w") as f:
        json.dump(detail, f, indent=2)
    print(f"Detailed parameter breakdown saved to: {detail_path}")


def main():
    parser = argparse.ArgumentParser(description="Profile Qwen2-VL model")
    parser.add_argument("--model", default="Qwen/Qwen2-VL-7B-Instruct",
                        help="Model name or path")
    parser.add_argument("--dry-run", action="store_true",
                        help="Config-only mode (no model download)")
    args = parser.parse_args()

    if args.dry_run:
        profile_config_only(args.model)
    else:
        profile_full_model(args.model)


if __name__ == "__main__":
    main()
