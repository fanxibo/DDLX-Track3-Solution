# Reproducibility Instructions

This document describes how to reproduce the inference pipeline and generate a Codabench-style submission zip.

## 1. Environment

```bash
conda create -n ddl_env python=3.10 -y
conda activate ddl_env
pip install -r requirements.txt
```

The final run was performed with:

- Python 3.10
- CUDA 12.4
- PyTorch 2.6.0
- Transformers 5.8.0
- 1x NVIDIA A30 24GB GPU

## 2. Model weights

The solution uses the public model weights from:

```text
Qwen/Qwen2-VL-7B-Instruct
```

The model is loaded automatically by `transformers` in `src/inference.py`. No fine-tuned checkpoints or private weights are required.

## 3. Dataset

Download the organizer-recommended DDL-X dataset and put test images under:

```text
./image/
```

Expected structure:

```text
DDLX-Track3-Solution/
├── image/
│   ├── 000001.jpg
│   ├── 000002.jpg
│   └── ...
├── src/
│   └── inference.py
└── requirements.txt
```

## 4. Run inference

```bash
python src/inference.py
```

By default, the script reads images from `./image/` and writes predictions to:

```text
submission.zip
```

Each prediction file is stored under the `json/` folder inside the zip.

## 5. Check submission format

```bash
python scripts/check_submission_format.py submission.zip
```

The checker validates that each JSON file contains the required fields:

- `classification_result`
- `bounding_boxes`
- `visible_forgery_traces`

For compatibility, it also accepts the human-readable field names used in the README:

- `Classification result`
- `Bounding boxes`
- `Visible forgery traces`

## 6. Final submitted file

The repository includes `results.zip`, which is the archived final prediction file corresponding to the submitted solution.
