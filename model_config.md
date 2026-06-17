# Model Configuration

## Model: Qwen2-VL-7B-Instruct

| Parameter | Value |
|-----------|-------|
| Model | Qwen/Qwen2-VL-7B-Instruct |
| Source | Alibaba Qwen Team (Apache 2.0) |
| Parameters | ~7.6 billion |
| Architecture | Vision Transformer + Qwen2 LLM |
| Weight format | bfloat16 (~14.3GB) |
| HuggingFace | https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct |

## Hardware

| Component | Spec |
|-----------|------|
| GPU | 1x NVIDIA A30 (24GB VRAM) |
| CUDA | 12.4 |
| PyTorch | 2.6.0+cu124 |
| Transformers | 5.8.0 |

## Inference Parameters

```python
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

model.generate(
    **inputs,
    max_new_tokens=200,       # output token limit
    do_sample=False,          # greedy decoding
    use_cache=True            # KV-cache enabled
)
```

## Download Model Weights

```python
# International
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen2-VL-7B-Instruct")

# China mirror
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
snapshot_download("Qwen/Qwen2-VL-7B-Instruct")
```
