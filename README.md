# DDL-X Track 3: Reconstructed Early Submission Version

This repository contains a reconstructed and organizer-rerunnable version of our early submission pipeline for the IJCAI 2026 DDL-X Track 3 challenge.

## Purpose

This repository is intended for reproducibility and organizer verification. It is not presented as a later improved prompt variant or a post-hoc optimized public release. Instead, it is the recovered inference path prepared to reproduce the early submission-stage behavior as closely as possible with public components and clear execution instructions.

## What this repository provides

This repository is designed to do one thing clearly:

* load one public open-source backbone model;
* run inference on the challenge images;
* generate one `results.zip` file;
* keep the challenge-required JSON schema;
* avoid personal absolute paths and machine-specific assumptions.

## Repository structure

```text
DDLX-Track3-Solution/
├── README.md
├── REPRODUCE.md
├── model_config.md
├── requirements.txt
├── run.sh
├── results.zip
└── src/
    └── inference.py
```

`results.zip` is the reference prediction archive included in the repository. Organizers can also regenerate it by following the instructions in `REPRODUCE.md`.

## Backbone model

Base model:

```text
Qwen/Qwen2-VL-7B-Instruct
```

Pinned revision used for reproducibility:

```text
eed13092ef92e448dd6875b2a00151bd3f7db0ac
```

## Input and output

Expected input image directory:

```text
./image/
```

Expected output archive:

```text
./results.zip
```

Inside the zip, predictions are written as:

```text
json/<image_stem>.json
```

## Output JSON schema

Each prediction JSON contains the following keys:

```json
{
  "Bounding boxes": [[x1, y1, x2, y2]] or "None",
  "Visible forgery traces": "text description",
  "Classification result": "fake" or "real"
}
```

## Quick start

```bash
conda create -n ddl_env python=3.10 -y
conda activate ddl_env
pip install -r requirements.txt
bash run.sh ./image ./results.zip
```

## Important note

If this repository is used for official organizer verification, `src/inference.py`, `REPRODUCE.md`, and the generated `results.zip` should remain consistent with each other. The documentation and code in this repository are written to describe the same reproducible inference path.
