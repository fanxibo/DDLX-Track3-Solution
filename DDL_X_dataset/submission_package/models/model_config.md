# Model Configuration

## Model: Qwen2-VL-7B-Instruct

- **Source**: Alibaba Qwen Team
- **Repository**: https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct
- **License**: Apache 2.0
- **Parameters**: ~7 billion
- **Architecture**: Vision-Language Model (VLM) with Vision Transformer + Qwen2 LLM

## Hardware Configuration

| Parameter | Value |
|-----------|-------|
| GPU | 1x NVIDIA A30 (24GB VRAM) |
| CUDA Version | 12.4 |
| PyTorch | 2.6.0+cu124 |
| Transformers | 5.8.0 |
| qwen-vl-utils | 0.0.14 |

## Inference Configuration

```python
# Model loading
Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    local_files_only=True
)

# Generation parameters
model.generate(
    **inputs,
    max_new_tokens=200,       # Maximum output tokens per image
    do_sample=False,          # Deterministic (greedy) decoding
    use_cache=True            # KV-cache for speed
)
```

## Model Weights

The model weights are automatically downloaded by the HuggingFace transformers library.
No local weight files are included in this submission (per competition rules allowing
open-source pre-trained models).

### Download Command

```python
from huggingface_hub import snapshot_download

# From HuggingFace (international)
snapshot_download("Qwen/Qwen2-VL-7B-Instruct")

# From HF Mirror (China)
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
snapshot_download("Qwen/Qwen2-VL-7B-Instruct")
```

### Model Size
- Total parameters: 7.6B
- Weight file size: ~14.3GB (bfloat16)
- Model files: 730 safetensors shards

## Prompt Template

### Version 1 (Baseline)
```
You are an expert deepfake forensic analyst. Analyze this image. 
Is it a real image or a fake/manipulated image? 
If fake, identify the bounding box of the manipulated region in [xmin, ymin, xmax, ymax] format 
(scaled from 1 to 1000) and describe the visible forgery traces.
Respond STRICTLY with a valid JSON object...
```

### Version 2 (Optimized, Final)
```
Deepfake forensic analysis. Is this image real or AI-generated/manipulated? 
Write specific forensic observations (anatomical errors, lighting inconsistencies, 
texture anomalies, edge artifacts, AI traces, physical implausibility). 
Reference exact locations.

For fake: TIGHT [[x1,y1,x2,y2]] box (1-1000) covering only the tampered region. 
For real: "None".

Output raw JSON: {"Classification result": "real", "Bounding boxes": "None", 
"Visible forgery traces": "your analysis"}
```

## Output Format Validation

The `validate()` function performs:
1. Flexible key matching (case/space/underscore insensitive)
2. Classification normalization ("fake"/"real")
3. Bounding box format conversion and clamping to [1, 1000]
4. Oversized box shrinking (>80% of image → shrink by 100px margin)
5. Array-to-string conversion for traces fields
6. Default fallback for failed outputs

## Reproducibility Notes

- `do_sample=False` ensures deterministic output for same input
- Vision processing uses default Qwen2-VL settings (448x448 resize, etc.)
- Image loading with `LOAD_TRUNCATED_IMAGES = True` for robustness
- Environment variable `HF_ENDPOINT` for China-based mirror access
