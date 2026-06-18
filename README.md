# DDL-X Track 3: Deepfake Detection, Localization, and Explainability

**Competition**: IJCAI 2026 AI Safety Workshop
**Track**: Track 3: Deepfake Detection, Localization, and Explainability
**Team**: fanxibo
**Version**: 2, Forensic Chain-of-Thought
**Solution type**: Zero-shot / inference-only vision-language model solution

## Overview

This repository provides our final solution for DDL-X Track 3. The task requires a system to perform three subtasks at the same time: deepfake classification, tampered-region localization, and visible forgery trace explanation.

Our solution uses **Qwen2-VL-7B-Instruct**, an open-source 7B vision-language model, in a zero-shot inference setting. For each input image, the model performs a single forward pass and outputs a structured prediction containing the classification result, bounding boxes, and forensic explanation.

The core idea of this solution is **Forensic Chain-of-Thought prompt engineering**. Instead of directly asking the model whether an image is real or fake, the prompt guides the model to inspect multiple types of forensic evidence, including facial anatomy, lighting consistency, texture artifacts, boundary artifacts, AI-generation traces, and physical plausibility. The model then produces a structured JSON-style answer for classification, localization, and explainability.

## Repository Structure

```text
DDLX-Track3-Solution/
├── README.md
├── LICENSE
├── requirements.txt
├── model_config.md
├── TRAINING.md
├── REPRODUCE.md
├── results.zip
├── src/
│   ├── inference.py
│   ├── download.py
│   └── fix_zip.py
└── scripts/
    └── check_submission_format.py
```

## File Description

`README.md` is the main project introduction file.

`LICENSE` provides the open-source license for the code in this repository.

`requirements.txt` records the Python dependencies required to run the inference pipeline.

`model_config.md` documents the model, hardware, and inference configuration used in the final solution.

`TRAINING.md` explains the training pipeline and hyperparameter configuration. Since this solution is zero-shot and inference-only, it does not include task-specific fine-tuning, LoRA training, or additional trainable parameters.

`REPRODUCE.md` provides step-by-step instructions for reproducing the inference pipeline and generating a Codabench-style submission file.

`results.zip` is the archived final prediction file corresponding to the submitted solution.

`src/inference.py` is the main inference script.

`src/download.py` is a dataset download helper. It does not contain hard-coded private tokens and reads the ModelScope token from the environment variable `MODELSCOPE_TOKEN` when needed.

`src/fix_zip.py` is a ZIP repair and format conversion utility.

`scripts/check_submission_format.py` checks whether the generated submission zip file follows the required JSON format.

## Approach

The solution performs zero-shot multi-task deepfake analysis with Qwen2-VL-7B-Instruct. For each image, the model is prompted to produce three outputs:

1. whether the image is real or fake;
2. where the manipulated region is located;
3. what visible traces support the decision.

The model is guided by a forensic prompt that asks it to examine the following evidence categories:

* facial anatomy and geometry;
* lighting and shadow consistency;
* skin texture and local artifacts;
* boundary and blending artifacts;
* AI-generation traces;
* physical and semantic plausibility.

The final output is parsed into a structured JSON prediction file.

## Key Design

### 1. Forensic Chain-of-Thought Prompt

The prompt structures the model's reasoning into three stages. First, the model inspects multiple forensic evidence categories. Second, it makes a real/fake decision based on the observed evidence. Third, it outputs a structured JSON object containing classification, localization, and explanation.

### 2. Robust Output Parser

The inference pipeline contains a robust output parser to reduce formatting errors from the vision-language model. It supports regex-based JSON extraction, markdown-wrapped JSON extraction, case-insensitive and space-insensitive key matching, automatic bounding-box format conversion, bounding-box validation, and classification fallback.

### 3. Streaming ZIP Generation and Crash Recovery

The predictions are written directly into a zip file. The pipeline supports resume-style execution by skipping completed entries when restarted. The utility script can also help repair corrupted or interrupted zip outputs.

## Environment Setup

Create a Python environment and install the dependencies:

```bash
conda create -n ddl_env python=3.10 -y
conda activate ddl_env
pip install -r requirements.txt
```

The final run was performed with the following environment:

```text
Python 3.10
CUDA 12.4
PyTorch 2.6.0
Transformers 5.8.0
GPU: NVIDIA A30 24GB
```

## Requirements

The main dependencies are listed in `requirements.txt`:

```text
torch==2.6.0
transformers==5.8.0
qwen-vl-utils
Pillow
tqdm
modelscope
huggingface_hub
```

## Model

The solution uses the public model weights of:

```text
Qwen/Qwen2-VL-7B-Instruct
```

The model is used as an open-source vision-language backbone. No task-specific fine-tuning is performed. No private model weights, LoRA adapters, or additional trainable checkpoints are used.

More details are provided in `model_config.md`.

## Data

The solution uses only the organizer-recommended DDL-X dataset. No external datasets, private datasets, or additional generated training samples are used.

Dataset sources:

* Hugging Face: `https://huggingface.co/datasets/zy23333/DDL-X`
* ModelScope: `https://modelscope.cn/datasets/DDLteam/DDL_X`

Expected local image directory:

```text
./image/
```

## Run Inference

Put the test images under:

```text
./image/
```

Then run:

```bash
python src/inference.py
```

Before running, check the `image_dir` and `zip_path` settings in `main()` if your local paths are different.

The script will generate a submission zip file containing one JSON prediction file for each image.

## Submission Format

The human-readable output format is:

```json
{
  "Classification result": "fake",
  "Bounding boxes": [[320, 180, 680, 820]],
  "Visible forgery traces": "The face region shows unnaturally smooth skin and inconsistent lighting around the boundary."
}
```

For Codabench-style format checking, the repository also supports the canonical lowercase field names:

```json
{
  "classification_result": "fake",
  "bounding_boxes": [[320, 180, 680, 820]],
  "visible_forgery_traces": "The face region shows unnaturally smooth skin and inconsistent lighting around the boundary."
}
```

Bounding-box coordinates are normalized to the range `[1, 1000]`.

## Check Submission Format

To validate the format of `results.zip`, run:

```bash
python scripts/check_submission_format.py results.zip
```

If the file format is valid, the script will print:

```text
Submission format looks valid.
```

## Training Pipeline and Hyperparameters

This solution does not perform task-specific training or fine-tuning. It is a zero-shot and inference-only solution based on the public Qwen2-VL-7B-Instruct model.

Therefore, there is no optimizer, learning rate schedule, training checkpoint, LoRA adapter, or task-specific model weight to release.

The main inference hyperparameters are:

```python
torch_dtype = torch.bfloat16
device_map = "auto"
max_new_tokens = 250
do_sample = False
use_cache = True
```

More details are provided in `TRAINING.md`.

## Reproducibility

The complete reproduction guide is provided in `REPRODUCE.md`.

In general, the reproduction process is:

```bash
conda create -n ddl_env python=3.10 -y
conda activate ddl_env
pip install -r requirements.txt
python src/inference.py
python scripts/check_submission_format.py results.zip
```

The final submitted prediction archive is provided as:

```text
results.zip
```

## Compliance Notes

This repository releases the complete implementation of our DDL-X Track 3 solution.

The solution is zero-shot and inference-only. It does not train or fine-tune the backbone model. Therefore, there is no task-specific training pipeline, optimizer, learning rate schedule, checkpoint, LoRA adapter, or additional trainable parameter to release.

The final model weights are the public `Qwen/Qwen2-VL-7B-Instruct` weights loaded through Hugging Face Transformers.

Only the organizer-recommended DDL-X dataset was used. No external datasets, private datasets, or additional generated training samples were used.

The final submission file is `results.zip`. The inference code is provided in `src/inference.py`. The model and hardware configuration are documented in `model_config.md`. The inference-only configuration is documented in `TRAINING.md`. Reproduction steps are provided in `REPRODUCE.md`.

## License

* Code: MIT License
* Model: Apache 2.0, Alibaba Qwen Team
* Data: Please refer to the official DDL-X dataset license and competition rules
