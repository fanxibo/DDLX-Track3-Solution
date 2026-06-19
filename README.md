# DDL-X Track 3: Reproduced early submission version

This repository contains the recovered and organizer-rerunnable version of our early test-phase submission for the IJCAI 2026 DDL-X Track 3 challenge.

## What this repository is for
This repo is **not** presented as a later improved prompt variant. It is the recovered code path that most likely corresponds to the submission period before the Codabench submissions on 2026-05-25 and 2026-05-26.

The goal of this repository is straightforward reproducibility:
- run one public open-source model;
- generate one `submission_test.zip` file;
- keep the same JSON schema required by the challenge;
- avoid hard-coded personal absolute paths.

## Repository structure
```text
DDLX-Track3-Solution/
├── README.md
├── REPRODUCE.md
├── model_config.md
├── requirements.txt
├── run.sh
└── src/
    └── inference.py
```

## Model
Backbone model:
- `Qwen/Qwen2-VL-7B-Instruct`

Recommended pinned revision for reproducibility:
- `eed13092ef92e448dd6875b2a00151bd3f7db0ac`

## Expected input and output
Input image directory:
```text
./image/
```

Output zip:
```text
submission_test.zip
```

Inside the zip, predictions are written as:
```text
json/<image_stem>.json
```

## Quick start
```bash
conda create -n ddl_env python=3.10 -y
conda activate ddl_env
pip install -r requirements.txt
bash run.sh ./image ./submission_test.zip
```

## Important note
If you want this repository to serve as the official verification repo, do **not** keep a later prompt variant in `src/inference.py`. The file should always match the recovered early-submission logic.
