"""
DDL-X Track 3: Deepfake Detection, Localization, and Explainability
Version 2 — Forensic Chain-of-Thought Prompt with Robust Output Parser
"""
import os, json, torch, re, zipfile
from PIL import Image, ImageFile
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ── Forensic Chain-of-Thought Prompt ──
PROMPT = """You are a forensic image analyst. Examine this image carefully to determine if it is a real photograph or an AI-generated/manipulated fake.

Step 1 — Look for specific forensic evidence:
- Anatomical errors: asymmetric eyes, distorted nose, irregular teeth, mismatched ears
- Lighting inconsistencies: conflicting shadow directions, inconsistent highlights
- Texture anomalies: unnaturally smooth skin, missing pores, hair merging into skin
- Edge artifacts: blending halos, cut-paste seams, sharpness discontinuities
- AI generation traces: GAN grid patterns, diffusion color bleed, checkerboard ghosting
- Physical implausibility: impossible reflections, perspective errors, floating parts

Step 2 — Decide: real photos have natural noise and subtle asymmetry. AI images often look "too perfect" or have specific localized compositing artifacts.
If fake, identify ONLY the specific tampered region with TIGHT bounding boxes, NOT the whole image.

Step 3 — Output ONLY raw JSON (no markdown):
{"Classification result": "real", "Bounding boxes": "None", "Visible forgery traces": "detailed analysis with specific spatial references"}

Now analyze:"""


def extract_json(text: str):
    """Extract JSON object from model output, handling markdown wrapping."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r'^```\w*\n?', '', text)
        text = re.sub(r'\n?```$', '', text)
    try:
        m = re.search(r'\{.*\}', text, re.DOTALL)
        return json.loads(m.group(0)) if m else None
    except Exception:
        return None


def validate(result: dict):
    """Robust output validator with key normalization, box refinement, and format fixing."""
    if result is None:
        return {"Bounding boxes": "None", "Visible forgery traces": "Analysis failed.",
                "Classification result": "real"}

    cls = boxes = traces = None
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

    # Normalize traces
    if traces is None or (isinstance(traces, str) and len(traces.strip()) < 20):
        traces = "No visible forgery traces detected."
    elif isinstance(traces, list):
        traces = "; ".join(str(t) for t in traces if str(t).strip())
    traces = str(traces)[:5000]

    # Normalize bounding boxes
    if cls == "real":
        fixed_boxes = "None"
    elif isinstance(boxes, str) and boxes.strip() != "None":
        try:
            boxes = json.loads(boxes)
        except:
            boxes = None

    if cls == "fake" and isinstance(boxes, list) and len(boxes) > 0:
        if isinstance(boxes[0], list):
            valid = []
            for b in boxes:
                if len(b) == 4 and all(isinstance(v, (int, float)) for v in b):
                    clamped = [max(1, min(1000, int(round(v)))) for v in b]
                    w, h = clamped[2] - clamped[0], clamped[3] - clamped[1]
                    # Shrink oversized boxes (>80% image)
                    if w > 800 and h > 800:
                        clamped = [clamped[0]+100, clamped[1]+100,
                                   clamped[2]-100, clamped[3]-100]
                    valid.append([max(1, min(1000, v)) for v in clamped])
            fixed_boxes = valid if valid else "None"
        elif len(boxes) == 4 and all(isinstance(v, (int, float)) for v in boxes):
            clamped = [max(1, min(1000, int(round(v)))) for v in boxes]
            w, h = clamped[2] - clamped[0], clamped[3] - clamped[1]
            if w > 800 and h > 800:
                clamped = [clamped[0]+100, clamped[1]+100,
                           clamped[2]-100, clamped[3]-100]
            fixed_boxes = [[max(1, min(1000, v)) for v in clamped]]
        else:
            fixed_boxes = "None"
        if fixed_boxes == "None":
            cls = "real"
    else:
        fixed_boxes = "None"
        if cls == "fake":
            cls = "real"

    return {"Bounding boxes": fixed_boxes, "Visible forgery traces": traces,
            "Classification result": cls}


def main():
    image_dir = "./image"          # path to test images
    zip_path = "submission.zip"    # output zip

    print("Loading Qwen2-VL-7B-Instruct...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=False       # set True after first download
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

    image_files = sorted(f for f in os.listdir(image_dir)
                         if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp')))
    print(f"Found {len(image_files)} test images.")

    # Resume support
    completed = set()
    if os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, 'r') as zf:
            completed = set(zf.namelist())
        print(f"Resuming: {len(completed)} already done.")

    with zipfile.ZipFile(zip_path, 'a', zipfile.ZIP_DEFLATED) as zf:
        for img_name in tqdm(image_files, desc="Inference"):
            json_name = f"json/{os.path.splitext(img_name)[0]}.json"
            if json_name in completed:
                continue
            try:
                img_path = os.path.join(image_dir, img_name)
                messages = [{"role": "user", "content": [
                    {"type": "image", "image": img_path},
                    {"type": "text", "text": PROMPT}]}]
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = processor(
                    text=[text], images=image_inputs, videos=video_inputs,
                    padding=True, return_tensors="pt").to("cuda")

                with torch.no_grad():
                    gen = model.generate(**inputs, max_new_tokens=250,
                                         do_sample=False, use_cache=True)
                output = processor.batch_decode(
                    gen[:, inputs.input_ids.shape[1]:],
                    skip_special_tokens=True)[0]
                result = validate(extract_json(output))

            except Exception as e:
                print(f"\n[Err] {img_name}: {e}")
                result = {"Bounding boxes": "None",
                          "Visible forgery traces": f"Error: {e}"[:200],
                          "Classification result": "real"}

            zf.writestr(json_name, json.dumps(result, ensure_ascii=False, indent=2))
            completed.add(json_name)

    print(f"\nDone! {len(completed)} predictions in {zip_path}")


if __name__ == "__main__":
    main()
