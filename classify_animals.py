"""
Zero-shot 動物圖片分類 — 使用剪枝後的 Qwen2-VL model

Usage:
    # 單張圖片
    python classify_animals.py --image photo.jpg

    # 整個資料夾
    python classify_animals.py --image-dir ./animals/

    # 指定候選類別
    python classify_animals.py --image photo.jpg --classes cat,dog,bird,rabbit,fish

    # 輸出 CSV
    python classify_animals.py --image-dir ./animals/ --output results.csv
"""

import argparse
import csv
import torch
from pathlib import Path
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, AutoConfig


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


def load_model(model_dir="pruned_model/"):
    model_dir = Path(model_dir)
    print(f"Loading model from: {model_dir}")

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
    return model, processor, device


def classify_image(model, processor, device, image_path, classes=None):
    img = Image.open(image_path).convert("RGB")

    if classes:
        class_list = ", ".join(classes)
        prompt = (
            f"Classify the animal in this image. "
            f"Choose exactly one from: [{class_list}]. "
            f"Reply with only the class name, nothing else."
        )
    else:
        prompt = "What animal is in this image? Reply with only the animal name, nothing else."

    messages = [{"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text", "text": prompt},
    ]}]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[text], images=[img], padding=True, return_tensors="pt")
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=20)

    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    label = processor.decode(generated, skip_special_tokens=True).strip()

    # If classes provided, match to closest class name
    if classes:
        label_lower = label.lower()
        for c in classes:
            if c.lower() in label_lower:
                return c
    return label


def main():
    parser = argparse.ArgumentParser(description="Zero-shot animal classification with pruned Qwen2-VL")
    parser.add_argument("--model", default="pruned_model/")
    parser.add_argument("--image", default=None, help="Single image path")
    parser.add_argument("--image-dir", default=None, help="Directory of images")
    parser.add_argument("--classes", default=None,
                        help="Comma-separated class names (e.g. cat,dog,bird)")
    parser.add_argument("--output", default=None, help="Output CSV path")
    args = parser.parse_args()

    if not args.image and not args.image_dir:
        parser.error("Provide --image or --image-dir")

    classes = [c.strip() for c in args.classes.split(",")] if args.classes else None
    model, processor, device = load_model(args.model)

    # Collect image paths
    image_paths = []
    if args.image:
        image_paths.append(Path(args.image))
    if args.image_dir:
        d = Path(args.image_dir)
        image_paths.extend(
            sorted(p for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        )

    print(f"\nClassifying {len(image_paths)} image(s)...")
    if classes:
        print(f"Classes: {classes}")
    print()

    results = []
    for path in image_paths:
        label = classify_image(model, processor, device, path, classes)
        print(f"  {path.name:40s} → {label}")
        results.append({"file": str(path), "label": label})

    if args.output:
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["file", "label"])
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
