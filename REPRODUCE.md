# Reproducibility Instructions

## 1. Environment
```bash
conda create -n ddl_env python=3.10 -y
conda activate ddl_env
pip install -r requirements.txt
```

Suggested runtime used in the public repo documentation:
- Python 3.10
- CUDA-capable GPU
- PyTorch 2.6.0
- Transformers 5.8.0

## 2. Model weights
This solution uses the public open-source backbone:
```text
Qwen/Qwen2-VL-7B-Instruct
```

For reproducibility, it is recommended to pin the model revision to:
```text
eed13092ef92e448dd6875b2a00151bd3f7db0ac
```

## 3. Dataset layout
Place the challenge test images under:
```text
./image/
```

Example:
```text
DDLX-Track3-Solution/
├── image/
│   ├── 000001.jpg
│   ├── 000002.jpg
│   └── ...
├── run.sh
├── requirements.txt
└── src/
    └── inference.py
```

## 4. Run inference
```bash
bash run.sh ./image ./submission_test.zip
```

Equivalent direct command:
```bash
python src/inference.py --image_dir ./image --output_zip ./submission_test.zip
```

If the model has already been downloaded locally, you can use:
```bash
python src/inference.py --image_dir ./image --output_zip ./submission_test.zip --local_files_only
```

## 5. Output format
The output zip must contain a `json/` directory, with one JSON file per image.
Each JSON must include:
- `Bounding boxes`
- `Visible forgery traces`
- `Classification result`

## 6. Verification advice
Before publishing or sending the repo to organizers:
1. regenerate `submission_test.zip` from this exact repo;
2. confirm the zip structure is `json/*.json`;
3. confirm the repo README does not describe a later prompt variant;
4. confirm no personal absolute paths remain in code or docs.
