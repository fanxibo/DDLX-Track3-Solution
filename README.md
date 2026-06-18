# DDL-X Track 3: Deepfake Detection, Localization, and Explainability

**Competition**: IJCAI 2026 AI Safety Workshop  
**Team**: fanxibo  
**Version**: 2 (Forensic Chain-of-Thought)

## Approach

Zero-shot multi-task deepfake analysis using **Qwen2-VL-7B-Instruct**, a 7B open-source vision-language model. A single forward pass simultaneously outputs classification (real/fake), tampered region localization (bounding box), and forensic explanation (text).

**Core innovation**: Forensic Chain-of-Thought prompt engineering — the model is guided through six categories of forensic evidence (anatomy, lighting, texture, edges, AI artifacts, physics) before making a decision, then outputs all three tasks in a single structured JSON.

## Files

```
├── README.md
├── model_config.md        # Model weights & hardware configuration
├── results.zip            # Submission predictions (100k images)
├── src/
│   ├── inference.py       # Main inference pipeline (V2)
│   ├── download.py        # Dataset download helper
│   └── fix_zip.py         # ZIP repair utility
```

## Key Design

### 1. Forensic Chain-of-Thought Prompt
The prompt structures the model's reasoning into three steps:
1. Examine six categories of forensic evidence
2. Make a decision based on evidence
3. Output structured JSON

### 2. Robust Output Parser
- Regex-based JSON extraction (handles markdown wrapping)
- Case/space/underscore-insensitive key matching
- Automatic box format conversion (list, string, nested)
- Oversized box shrinking (>80% image → auto-shrink by 100px margins)
- Classification fallback and validation

### 3. Streaming ZIP with Crash Recovery
- Results written directly to zip (no intermediate files)
- Resume support: completed entries are skipped on restart
- Binary recovery: corrupted zip entries extracted by scanning local file headers

## Environment

```bash
conda create -n ddl_env python=3.10 -y && conda activate ddl_env
pip install torch==2.6.0 transformers==5.8.0 qwen-vl-utils pillow tqdm
```

## Model

Qwen2-VL-7B-Instruct (Apache 2.0) — see `model_config.md` for download and configuration.

## Data

10,0000 test images from DDL-X dataset:
- HuggingFace: https://huggingface.co/datasets/zy23333/DDL-X
- ModelScope: https://modelscope.cn/datasets/DDLteam/DDL_X

## Run

```bash
python src/inference.py
```

Edit `image_dir` and `zip_path` in `main()` before running. Supports GPU auto-detection via `device_map="auto"`.

## Submission Format

```json
{
  "Classification result": "fake",
  "Bounding boxes": [[320, 180, 680, 820]],
  "Visible forgery traces": "The face region shows unnaturally smooth skin..."
}
```

## License

- Code: MIT
- Model: Apache 2.0 (Alibaba Qwen Team)
- Data: Apache 2.0
