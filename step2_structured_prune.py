"""
Step 2: 結構化剪枝 — ShortGPT-style Layer Removal for Qwen2-VL
===============================================================
用 Block Influence (BI) score 找出最不重要的 layers 並移除。

Qwen2-VL 優勢：無 cross-attention，任何 LM layer 都可以安全移除。

Usage:
    python step2_structured_prune.py --model Qwen/Qwen2-VL-7B-Instruct \
                                     --prune-ratio 0.2 \
                                     --calibration-size 32 \
                                     --output pruned_model/
"""

import argparse
import json
import torch
import torch.nn.functional as F
from pathlib import Path


def load_model(model_name):
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

    print(f"Loading model: {model_name}")
    processor = AutoProcessor.from_pretrained(model_name)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map=None,
    ).cpu()
    model.eval()
    return model, processor


def prepare_calibration_data(processor, n_samples=32):
    """準備校準資料（文字 + 圖文混合）。"""
    from PIL import Image
    import numpy as np

    samples = []

    # Text-only samples
    text_prompts = [
        "Explain the concept of neural network pruning.",
        "What is the capital of France?",
        "Describe how a transformer model works.",
        "Write a Python function to sort a list.",
        "What are the benefits of quantization in deep learning?",
        "Explain the difference between structured and unstructured pruning.",
        "How does attention mechanism work in transformers?",
        "What is knowledge distillation?",
        "Describe the architecture of a vision transformer.",
        "What is compute-in-memory for AI acceleration?",
        "Explain grouped query attention.",
        "How does rotary position embedding work?",
        "What is the difference between INT4 and INT8 quantization?",
        "Describe the SwiGLU activation function.",
        "How does model compression work for edge devices?",
        "What is the role of layer normalization?",
    ]

    for prompt in text_prompts[:n_samples // 2]:
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], padding=True, return_tensors="pt")
        samples.append(("text", inputs))

    # Image-text samples (synthetic images for calibration)
    image_prompts = [
        "Describe this image in detail.",
        "What objects can you see?",
        "What is the main subject?",
        "Describe the colors.",
        "Is there any text in this image?",
        "What is happening in this image?",
        "Describe the layout of this image.",
        "What patterns do you notice?",
    ]

    for prompt in image_prompts[:n_samples // 2]:
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        messages = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": prompt},
        ]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[img], padding=True, return_tensors="pt")
        samples.append(("image", inputs))

    print(f"Prepared {len(samples)} calibration samples "
          f"({sum(1 for t,_ in samples if t=='text')} text, "
          f"{sum(1 for t,_ in samples if t=='image')} image)")
    return samples


def get_lm_layers(model):
    """取得 Qwen2-VL 的 language model layers。"""
    # Qwen2VL: model.model.language_model.layers
    if hasattr(model, 'model') and hasattr(model.model, 'language_model'):
        return model.model.language_model.layers
    # fallback: model.language_model.layers
    if hasattr(model, 'language_model') and hasattr(model.language_model, 'layers'):
        return model.language_model.layers
    # fallback: model.model.layers
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers
    raise AttributeError("Cannot find LM layers in model. Check model structure.")


def compute_block_influence(model, calibration_samples, device):
    """
    計算每個 decoder layer 的 Block Influence (BI) score。
    BI = 1 - cosine_similarity(layer_input, layer_output)
    BI 越低 = 該層改變越少 = 越可以被移除。
    """
    print("\nComputing Block Influence scores...")

    layers = get_lm_layers(model)
    n_layers = len(layers)
    bi_scores = torch.zeros(n_layers)
    n_valid = 0

    # Register hooks
    hidden_before = {}
    hidden_after = {}
    hooks = []

    def make_pre_hook(idx):
        def hook(module, args):
            h = args[0] if isinstance(args[0], torch.Tensor) else args[0][0]
            hidden_before[idx] = h.detach().float().cpu()
        return hook

    def make_post_hook(idx):
        def hook(module, args, output):
            h = output[0] if isinstance(output, tuple) else output
            hidden_after[idx] = h.detach().float().cpu()
        return hook

    for i, layer in enumerate(layers):
        hooks.append(layer.register_forward_pre_hook(make_pre_hook(i)))
        hooks.append(layer.register_forward_hook(make_post_hook(i)))

    # Run calibration
    with torch.no_grad():
        for sample_type, sample in calibration_samples:
            try:
                sample = {k: v.to(device) if hasattr(v, 'to') else v
                          for k, v in sample.items()}
                model(**sample)

                for i in range(n_layers):
                    if i in hidden_before and i in hidden_after:
                        before = hidden_before[i].flatten()
                        after = hidden_after[i].flatten()
                        cos_sim = F.cosine_similarity(
                            before.unsqueeze(0), after.unsqueeze(0))
                        bi_scores[i] += (1.0 - cos_sim.item())

                n_valid += 1
            except Exception as e:
                print(f"  Skipping {sample_type} sample: {e}")
            finally:
                hidden_before.clear()
                hidden_after.clear()

    for h in hooks:
        h.remove()

    if n_valid > 0:
        bi_scores /= n_valid
        print(f"  Computed BI scores from {n_valid} samples")

    return bi_scores


def identify_layers_to_remove(bi_scores, prune_ratio):
    """
    根據 BI scores 決定要移除哪些 layers。
    Qwen2-VL 無 cross-attention，只需保護首尾層。
    """
    n_layers = len(bi_scores)
    n_remove = int(n_layers * prune_ratio)

    # Protected: first and last layers only (no cross-attention to protect!)
    protected = {0, n_layers - 1}

    candidates = [(i, bi_scores[i].item()) for i in range(n_layers)
                  if i not in protected]
    candidates.sort(key=lambda x: x[1])  # ascending = least important first

    to_remove = [idx for idx, _ in candidates[:n_remove]]
    to_remove.sort()

    return to_remove, protected


def remove_layers(model, layers_to_remove):
    """移除指定的 decoder layers。"""
    layers = get_lm_layers(model)
    keep_indices = [i for i in range(len(layers)) if i not in layers_to_remove]
    new_layers = torch.nn.ModuleList([layers[i] for i in keep_indices])

    # Set layers back to correct location
    if hasattr(model, 'model') and hasattr(model.model, 'language_model'):
        model.model.language_model.layers = new_layers
    elif hasattr(model, 'language_model'):
        model.language_model.layers = new_layers
    else:
        model.model.layers = new_layers

    if hasattr(model.config, 'text_config'):
        model.config.text_config.num_hidden_layers = len(new_layers)
    else:
        model.config.num_hidden_layers = len(new_layers)
    return model, keep_indices


def main():
    parser = argparse.ArgumentParser(
        description="Structured pruning for Qwen2-VL via ShortGPT BI scores")
    parser.add_argument("--model", default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--prune-ratio", type=float, default=0.2,
                        help="Fraction of LM layers to remove (default: 0.2 = 20%%)")
    parser.add_argument("--calibration-size", type=int, default=32)
    parser.add_argument("--output", default="pruned_model/")
    parser.add_argument("--scores-only", action="store_true",
                        help="Only compute BI scores, don't prune")
    args = parser.parse_args()

    model, processor = load_model(args.model)
    device = next(model.parameters()).device

    samples = prepare_calibration_data(processor, args.calibration_size)
    bi_scores = compute_block_influence(model, samples, device)

    # Print BI scores
    n_layers = len(bi_scores)
    print(f"\n{'=' * 60}")
    print(f"BLOCK INFLUENCE SCORES — Qwen2-VL ({n_layers} LM layers)")
    print(f"{'=' * 60}")
    print(f"  (lower BI = more redundant = safe to remove)")
    print(f"  Qwen2-VL 無 cross-attention，所有 LM layers 都可以被剪枝\n")

    # Vision encoder full-attention layers (for reference)
    full_attn_vis = {7, 15, 23, 31}

    for i, score in enumerate(bi_scores):
        bar = "█" * int(score.item() * 50)
        print(f"  Layer {i:3d}: {score.item():.4f} {bar}")

    # Save scores
    scores_path = Path(args.output) / "bi_scores.json"
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    with open(scores_path, "w") as f:
        json.dump({
            "model": args.model,
            "architecture": "Qwen2-VL (no cross-attention, M-RoPE fusion)",
            "num_lm_layers": n_layers,
            "bi_scores": {str(i): s.item() for i, s in enumerate(bi_scores)},
        }, f, indent=2)
    print(f"\nBI scores saved to: {scores_path}")

    if args.scores_only:
        print("\n[scores-only mode] Skipping pruning.")
        return

    to_remove, protected = identify_layers_to_remove(bi_scores, args.prune_ratio)

    print(f"\n{'=' * 60}")
    print(f"PRUNING PLAN")
    print(f"{'=' * 60}")
    print(f"  Total LM layers:  {n_layers}")
    print(f"  Protected layers: {sorted(protected)}")
    print(f"  Removing layers:  {to_remove} ({len(to_remove)} layers)")
    print(f"  Remaining layers: {n_layers - len(to_remove)}")

    model, kept = remove_layers(model, to_remove)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Layers kept: {kept}")
    print(f"  Remaining params: {total_params / 1e9:.2f}B")

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving pruned model to: {output_path}")
    # save_pretrained 在修改 Qwen2-VL layers 後會壞掉，手動存
    torch.save(model.state_dict(), output_path / "pytorch_model.bin")
    model.config.save_pretrained(output_path)
    processor.save_pretrained(output_path)
    print(f"  State dict saved ({sum(1 for _ in model.state_dict())} tensors)")

    meta = {
        "original_model": args.model,
        "architecture": "Qwen2-VL",
        "prune_ratio": args.prune_ratio,
        "original_lm_layers": n_layers,
        "removed_layers": to_remove,
        "kept_layers": kept,
        "remaining_lm_layers": len(kept),
        "remaining_params_B": total_params / 1e9,
        "bi_scores": {str(i): s.item() for i, s in enumerate(bi_scores)},
    }
    with open(output_path / "pruning_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("\nDone! Next step: run step3_quantize.py on the pruned model.")


if __name__ == "__main__":
    main()
