import argparse
import json
import os
import re
import zipfile

import torch
from PIL import ImageFile
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

ImageFile.LOAD_TRUNCATED_IMAGES = True
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

PROMPT = """You are a deepfake forensic expert. Analyze this image and output STRICT JSON (no markdown):
{
"Bounding boxes": [[x1,y1,x2,y2]] or "None",
"Visible forgery traces": "detailed forensic analysis of tampering artifacts or natural features",
"Classification result": "fake" or "real"
}
Coordinates scaled to 1-1000. Examine anatomical consistency, lighting and shadows, texture continuity, edge artifacts, AI generation traces, compression artifacts, and color coherence."""

def extract_json(text: str):
text = text.strip()
for prefix in ("`json", "`", "'''json", "'''"):
if text.startswith(prefix):
text = text[len(prefix):].strip()
for suffix in ("```", "'''"):
if text.endswith(suffix):
text = text[:-len(suffix)].strip()

```
try:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return json.loads(match.group(0)) if match else None
except Exception:
    return None
```

def validate(result):
if result is None:
return {
"Bounding boxes": "None",
"Visible forgery traces": "Analysis failed.",
"Classification result": "real",
}

```
cls = None
boxes = None
traces = None

for k, v in result.items():
    kl = k.lower().replace(" ", "").replace("_", "")
    if "classification" in kl:
        cls = str(v).lower().strip()
    elif "bounding" in kl:
        boxes = v
    elif "visible" in kl or "forgery" in kl:
        traces = v

if cls not in ("fake", "real"):
    cls = "real"

if traces is None:
    traces = "No visible forgery traces detected."
elif isinstance(traces, list):
    traces = "; ".join(str(t) for t in traces if str(t).strip())
else:
    traces = str(traces)

if not traces.strip():
    traces = "No visible forgery traces detected."

traces = traces[:3000]

if cls == "real":
    fixed_boxes = "None"
else:
    if isinstance(boxes, str) and boxes.strip() != "None":
        try:
            boxes = json.loads(boxes)
        except Exception:
            boxes = None

    if isinstance(boxes, list) and len(boxes) > 0:
        if isinstance(boxes[0], list):
            valid = []
            for b in boxes:
                if len(b) == 4 and all(isinstance(v, (int, float)) for v in b):
                    valid.append([max(1, min(1000, int(round(v)))) for v in b])
            fixed_boxes = valid if valid else "None"
        elif len(boxes) == 4 and all(isinstance(v, (int, float)) for v in boxes):
            fixed_boxes = [[max(1, min(1000, int(round(v)))) for v in boxes]]
        else:
            fixed_boxes = "None"
    else:
        fixed_boxes = "None"

    if fixed_boxes == "None":
        cls = "real"

return {
    "Bounding boxes": fixed_boxes,
    "Visible forgery traces": traces,
    "Classification result": cls,
}
```

def run_inference(
image_dir: str,
output_zip: str,
model_name: str,
revision: str,
local_files_only: bool,
):
print(f"Loading {model_name} (revision={revision})...")

```
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_name,
    revision=revision,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    local_files_only=local_files_only,
)
processor = AutoProcessor.from_pretrained(
    model_name,
    revision=revision,
    local_files_only=local_files_only,
)

image_files = sorted(
    f
    for f in os.listdir(image_dir)
    if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp"))
)
print(f"Found {len(image_files)} images.")

output_dir = os.path.dirname(os.path.abspath(output_zip))
if output_dir:
    os.makedirs(output_dir, exist_ok=True)

completed = set()
if os.path.exists(output_zip):
    with zipfile.ZipFile(output_zip, "r") as zf:
        completed = set(zf.namelist())
    print(f"Resuming: {len(completed)} entries already exist in {output_zip}.")

with zipfile.ZipFile(output_zip, "a", zipfile.ZIP_DEFLATED) as zf:
    for img_name in tqdm(image_files, desc="Inference"):
        json_name = f"json/{os.path.splitext(img_name)[0]}.json"
        if json_name in completed:
            continue

        try:
            img_path = os.path.join(image_dir, img_name)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img_path},
                        {"type": "text", "text": PROMPT},
                    ],
                }
            ]

            text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to("cuda")

            with torch.no_grad():
                gen = model.generate(
                    **inputs,
                    max_new_tokens=200,
                    do_sample=False,
                    use_cache=True,
                )

            output = processor.batch_decode(
                gen[:, inputs.input_ids.shape[1]:],
                skip_special_tokens=True,
            )[0]
            result = validate(extract_json(output))

        except Exception as e:
            print(f"[Err] {img_name}: {e}")
            result = {
                "Bounding boxes": "None",
                "Visible forgery traces": f"Error: {e}"[:200],
                "Classification result": "real",
            }

        zf.writestr(json_name, json.dumps(result, ensure_ascii=False, indent=2))
        completed.add(json_name)

print(f"Done! {len(completed)} predictions written to {output_zip}")
```

def main():
parser = argparse.ArgumentParser()
parser.add_argument("--image_dir", default="./image")
parser.add_argument("--output_zip", default="./results.zip")
parser.add_argument("--model_name", default="Qwen/Qwen2-VL-7B-Instruct")
parser.add_argument(
"--revision",
default="eed13092ef92e448dd6875b2a00151bd3f7db0ac",
)
parser.add_argument("--local_files_only", action="store_true")
args = parser.parse_args()

```
run_inference(
    image_dir=args.image_dir,
    output_zip=args.output_zip,
    model_name=args.model_name,
    revision=args.revision,
    local_files_only=args.local_files_only,
)
```

if **name** == "**main**":
main()
