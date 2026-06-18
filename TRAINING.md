# Training Pipeline and Hyperparameter Configuration

This solution does not perform task-specific training or fine-tuning.

The final competition submission is an inference-only / zero-shot solution based on the open-source backbone `Qwen/Qwen2-VL-7B-Instruct`. No additional external datasets were used. No extra generated samples, private data, or third-party training data were used.

## Backbone

- Model: Qwen/Qwen2-VL-7B-Instruct
- License: Apache 2.0
- Usage: zero-shot inference
- Fine-tuning: none
- LoRA / adapters: none
- Extra trainable parameters: none

## Inference hyperparameters

The inference configuration used in the final solution is:

```python
torch_dtype = torch.bfloat16
device_map = "auto"
max_new_tokens = 250
do_sample = False
use_cache = True
```

## Prompting strategy

The method uses a forensic Chain-of-Thought style prompt to guide the vision-language model to inspect facial manipulation evidence, including anatomical errors, lighting inconsistency, texture anomaly, edge artifact, AI-generation trace, and physical implausibility. The model is then required to output a structured JSON object containing classification, localization, and explanation.

## Dataset usage

Only the organizer-recommended DDL-X dataset was used for inference and evaluation. No external datasets were used.
