"""
Inference script for pruned Qwen2-VL model (FP32).

Usage:
    # Text-only
    python inference_pruned.py --prompt "什麼是剪枝？"

    # Image + text
    python inference_pruned.py --image photo.jpg --prompt "描述這張圖片"
"""

import argparse
import torch
from pathlib import Path
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, AutoConfig


def load_pruned_model(model_dir="pruned_model/"):
    model_dir = Path(model_dir)
    print(f"Loading pruned model from: {model_dir}")

    processor = AutoProcessor.from_pretrained(model_dir)
    config = AutoConfig.from_pretrained(model_dir)
    model = Qwen2VLForConditionalGeneration(config)

    state_dict = torch.load(
        model_dir / "pytorch_model.bin", map_location="cpu", weights_only=True
    )
    model.load_state_dict(state_dict)
    model.eval()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device)
    print(f"  Device: {device}")
    print(f"  Params: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    return model, processor, device


def generate(model, processor, device, prompt, image_path=None, max_new_tokens=256):
    from PIL import Image

    content = []
    images = []

    if image_path:
        img = Image.open(image_path).convert("RGB")
        content.append({"type": "image", "image": img})
        images.append(img)

    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    if images:
        inputs = processor(text=[text], images=images, padding=True, return_tensors="pt")
    else:
        inputs = processor(text=[text], padding=True, return_tensors="pt")

    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    # Decode only the generated part
    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    return processor.decode(generated, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="Inference with pruned Qwen2-VL")
    parser.add_argument("--model", default="pruned_model/")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--image", default=None, help="Path to image file")
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()

    model, processor, device = load_pruned_model(args.model)
    response = generate(model, processor, device, args.prompt, args.image, args.max_tokens)
    print(f"\n{'='*60}")
    print(f"Prompt: {args.prompt}")
    if args.image:
        print(f"Image:  {args.image}")
    print(f"{'='*60}")
    print(response)


if __name__ == "__main__":
    main()
