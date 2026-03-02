"""
Zero-shot 動物圖片分類 — 使用剪枝後的 Qwen2-VL model

Usage:
    # 單張圖片
    python classify_animals.py --image photo.jpg

    # 指定候選類別
    python classify_animals.py --image photo.jpg --classes cat,dog,lion,elephant

    # ImageFolder 格式 dataset（自動算 accuracy）
    # dataset/val/{cat,dog,lion,elephant}/*.jpg
    python classify_animals.py --dataset dataset/val

    # 跑 train + val 一起
    python classify_animals.py --dataset dataset/val --output results.csv

    # 整個資料夾（無 label，純分類）
    python classify_animals.py --image-dir ./photos/
"""

import argparse
import csv
import torch
from collections import defaultdict
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
    if not torch.backends.mps.is_available() and torch.cuda.is_available():
        device = "cuda"
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

    if classes:
        label_lower = label.lower()
        for c in classes:
            if c.lower() in label_lower:
                return c
    return label


def collect_dataset(dataset_dir):
    """Collect images from ImageFolder structure: dataset_dir/{class_name}/*.jpg"""
    dataset_dir = Path(dataset_dir)
    samples = []  # (image_path, ground_truth_label)
    classes = sorted(
        d.name for d in dataset_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )
    for cls in classes:
        cls_dir = dataset_dir / cls
        for img_path in sorted(cls_dir.iterdir()):
            if img_path.suffix.lower() in IMAGE_EXTS:
                samples.append((img_path, cls))
    return samples, classes


def print_report(results, classes):
    """Print accuracy report with per-class breakdown and confusion matrix."""
    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    accuracy = correct / total if total else 0

    print(f"\n{'='*60}")
    print(f"CLASSIFICATION REPORT")
    print(f"{'='*60}")
    print(f"  Total images:  {total}")
    print(f"  Correct:       {correct}")
    print(f"  Accuracy:      {accuracy:.1%}")

    # Per-class accuracy
    per_class = defaultdict(lambda: {"total": 0, "correct": 0})
    for r in results:
        gt = r["ground_truth"]
        per_class[gt]["total"] += 1
        if r["correct"]:
            per_class[gt]["correct"] += 1

    print(f"\n  {'Class':<15} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    print(f"  {'-'*43}")
    for cls in classes:
        c = per_class[cls]
        acc = c["correct"] / c["total"] if c["total"] else 0
        print(f"  {cls:<15} {c['correct']:>8} {c['total']:>8} {acc:>9.1%}")

    # Confusion matrix
    print(f"\n  Confusion Matrix (rows=actual, cols=predicted):")
    confusion = defaultdict(lambda: defaultdict(int))
    for r in results:
        confusion[r["ground_truth"]][r["predicted"]] += 1

    # Collect all predicted labels (including unexpected ones)
    all_labels = list(classes)
    for r in results:
        if r["predicted"] not in all_labels:
            all_labels.append(r["predicted"])

    header = f"  {'':>12}" + "".join(f"{c:>10}" for c in all_labels)
    print(header)
    for gt in classes:
        row = f"  {gt:>12}"
        for pred in all_labels:
            count = confusion[gt][pred]
            row += f"{count:>10}" if count else f"{'·':>10}"

        print(row)

    return accuracy


def main():
    parser = argparse.ArgumentParser(
        description="Zero-shot animal classification with pruned Qwen2-VL")
    parser.add_argument("--model", default="pruned_model/")
    parser.add_argument("--image", default=None, help="Single image path")
    parser.add_argument("--image-dir", default=None, help="Flat directory of images")
    parser.add_argument("--dataset", default=None,
                        help="ImageFolder path: dataset/{class_name}/*.jpg")
    parser.add_argument("--classes", default=None,
                        help="Comma-separated class names (e.g. cat,dog,lion,elephant)")
    parser.add_argument("--output", default=None, help="Output CSV path")
    args = parser.parse_args()

    if not args.image and not args.image_dir and not args.dataset:
        parser.error("Provide --image, --image-dir, or --dataset")

    model, processor, device = load_model(args.model)

    # === ImageFolder dataset mode (with ground truth) ===
    if args.dataset:
        samples, discovered_classes = collect_dataset(args.dataset)
        classes = ([c.strip() for c in args.classes.split(",")]
                   if args.classes else discovered_classes)

        print(f"\nDataset: {args.dataset}")
        print(f"Classes: {classes}")
        print(f"Images:  {len(samples)}")
        print()

        results = []
        for i, (path, gt) in enumerate(samples):
            pred = classify_image(model, processor, device, path, classes)
            is_correct = pred.lower() == gt.lower()
            mark = "OK" if is_correct else "X"
            print(f"  [{i+1}/{len(samples)}] {path.parent.name}/{path.name:30s} "
                  f"→ {pred:<12} (gt: {gt}) [{mark}]")
            results.append({
                "file": str(path),
                "ground_truth": gt,
                "predicted": pred,
                "correct": is_correct,
            })

        print_report(results, classes)

        if args.output:
            with open(args.output, "w", newline="") as f:
                writer = csv.DictWriter(
                    f, fieldnames=["file", "ground_truth", "predicted", "correct"])
                writer.writeheader()
                writer.writerows(results)
            print(f"\nResults saved to: {args.output}")
        return

    # === Single image / flat directory mode (no ground truth) ===
    classes = [c.strip() for c in args.classes.split(",")] if args.classes else None

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
