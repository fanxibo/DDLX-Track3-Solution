# DDL-X Track 3: Deepfake Detection, Localization, and Explainability

**Competition**: IJCAI 2026 AI Safety Workshop — DDL 2.0  
**Team**: fanxibo  
**Codabench**: https://www.codabench.org/competitions/15686/

## Approach

Zero-shot inference using **Qwen2-VL-7B-Instruct**, a 7B vision-language model, to perform all three tasks in a single forward pass:

| Task | Method | Metric |
|------|--------|--------|
| Detection | VLM classifies real vs fake | ACC |
| Localization | VLM outputs bounding box of tampered region | IoU |
| Explainability | VLM generates forensic analysis text | BERTScore |

No fine-tuning. Uses forensic prompt engineering to guide the model.

## Files

```
├── README.md              # This file
├── model_config.md        # Model weights and configuration
├── results.zip            # Final predictions (100k images)
├── src/
│   ├── inference.py       # Main inference script (V1)
│   ├── download.py        # Dataset download helper
│   └── fix_zip.py         # ZIP repair utility
```

## Environment

```bash
conda create -n ddl_env python=3.10 -y
conda activate ddl_env
pip install torch==2.6.0 transformers==5.8.0 qwen-vl-utils pillow tqdm
```

## Data

Test images should be in the structure: `image/*.png`, `image/*.jpg`

Dataset sources:
- HuggingFace: https://huggingface.co/datasets/zy23333/DDL-X
- ModelScope: https://modelscope.cn/datasets/DDLteam/DDL_X

## Run Inference

```bash
python src/inference.py
```

Configurable in `inference.py`:
- `image_dir`: path to test images
- `zip_path`: output zip path
- `max_new_tokens`: generation length (default 200)

Supports resume — restarting skips already-processed images.

## Submission Format

Each prediction is `json/{md5}.json` in `results.zip`:

```json
{
  "Classification result": "fake",
  "Bounding boxes": [[320, 180, 680, 820]],
  "Visible forgery traces": "The face region shows unnaturally smooth skin..."
}
```

- Coordinates in 1-1000 scale
- `"Bounding boxes"`: `"None"` for real, `[[x1,y1,x2,y2]]` for fake
- `"Visible forgery traces"`: detailed forensic paragraph

## License

- Qwen2-VL-7B: Apache 2.0 (Alibaba)
- Dataset: Apache 2.0
- Our code: MIT
