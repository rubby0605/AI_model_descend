# AI Model Descend

Qwen2-VL-7B → Edge AI (ASIC/CIM) 模型壓縮實驗。

## Why Qwen2-VL?
- Apache 2.0 開源，不需要 Meta 授權
- 無 cross-attention（用 M-RoPE fusion），剪枝比 LLaMA Vision 簡單
- 7B 比 LLaMA 11B 小，更容易實驗
- 架構：ViT encoder → PatchMerger → Qwen2 LM

## Pipeline

```
step1_profile.py      分析模型架構、各 component 參數量
        ↓
step2_structured_prune.py   ShortGPT BI score 結構化剪枝
        ↓
step3_quantize.py     AWQ/BnB 量化 (LM: INT4, Vision: INT8)
        ↓
ONNX export           → ASIC compiler toolchain
```

## Quick Start

```bash
# 1. 安裝
pip install -r requirements.txt

# 2. 先看模型架構（不用下載權重）
python step1_profile.py --dry-run

# 3. 完整 profiling（需要 GPU + ~16GB VRAM）
python step1_profile.py

# 4. 結構化剪枝（先只算 BI scores）
python step2_structured_prune.py --scores-only --output results/

# 5. 剪枝 20% (28→22 layers)
python step2_structured_prune.py --prune-ratio 0.2 --output pruned_model/

# 6. 量化（先看壓縮估算 + CIM mapping）
python step3_quantize.py --estimate-only

# 7. INT4 量化
python step3_quantize.py --model pruned_model/ --output quantized_model/
```

## ASIC/CIM Notes

- CIM 適合固定精度 MAC (INT4/INT8)，結構化剪枝讓矩陣維度對齊 CIM 陣列
- Vision encoder 對量化敏感，建議 INT8；Language model 用 INT4
- hidden_size=3584 = 256×14，完美對齊 256×256 CIM array
- 無 cross-attention = 比 LLaMA Vision 省大量 CIM tiles 和 data movement
- Target: ~5-10x compression, ~90% accuracy retention
