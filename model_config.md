# Model configuration

## Backbone
- Model: `Qwen/Qwen2-VL-7B-Instruct`
- Suggested pinned revision: `eed13092ef92e448dd6875b2a00151bd3f7db0ac`

## Inference settings
- dtype: `torch.bfloat16`
- device_map: `auto`
- generation: greedy decoding (`do_sample=False`)
- `max_new_tokens=200`

## Input / output
- input image directory: provided via `--image_dir`
- output archive: provided via `--output_zip`
- output JSON schema:
  - `Bounding boxes`
  - `Visible forgery traces`
  - `Classification result`

## Notes
This configuration file is intended to describe the recovered early-submission reproduction setup, not a later improved prompt variant.
