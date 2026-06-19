# Reproducibility Instructions

## 1. Environment setup

Create a clean Python environment and install the required packages:

```bash
conda create -n ddl_env python=3.10 -y
conda activate ddl_env
pip install -r requirements.txt
```

A CUDA-capable GPU is recommended for running this repository.

## 2. Model

This repository uses the public open-source backbone:

```text
Qwen/Qwen2-VL-7B-Instruct
```

For reproducibility, the recommended pinned revision is:

```text
eed13092ef92e448dd6875b2a00151bd3f7db0ac
```

## 3. Dataset layout

Place the challenge images in the following directory:

```text
./image/
```

Example repository layout:

```text
DDLX-Track3-Solution/
├── image/
│   ├── 000001.jpg
│   ├── 000002.jpg
│   └── ...
├── run.sh
├── results.zip
├── requirements.txt
└── src/
    └── inference.py
```

## 4. Run inference

Recommended command:

```bash
bash run.sh ./image ./results.zip
```

Equivalent direct command:

```bash
python src/inference.py --image_dir ./image --output_zip ./results.zip
```

If the model files have already been downloaded locally, you can run:

```bash
python src/inference.py --image_dir ./image --output_zip ./results.zip --local_files_only
```

If you want to explicitly pin the model revision at runtime, you can run:

```bash
python src/inference.py --image_dir ./image --output_zip ./results.zip --revision eed13092ef92e448dd6875b2a00151bd3f7db0ac
```

## 5. Output format

The output zip must contain a `json/` directory, with one JSON file per image:

```text
results.zip
└── json/
    ├── 000001.json
    ├── 000002.json
    └── ...
```

Each JSON file must include:

* `Bounding boxes`
* `Visible forgery traces`
* `Classification result`

## 6. Verification checklist

Before sending this repository to organizers, verify the following:

1. `src/inference.py` matches the intended reproducible inference logic.
2. The documented output filename is `results.zip`.
3. The generated zip structure is `json/*.json`.
4. No personal absolute paths remain in the code or documentation.
5. The model name and pinned revision are documented consistently.

## 7. Included artifact

The repository may include a `results.zip` file as a reference artifact. Organizers may either inspect that file directly or regenerate it from the provided code and instructions.
