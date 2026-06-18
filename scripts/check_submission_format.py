#!/usr/bin/env python3
import json
import sys
import zipfile
from pathlib import Path

CANONICAL_KEYS = {
    "classification_result",
    "bounding_boxes",
    "visible_forgery_traces",
}

ALIASES = {
    "Classification result": "classification_result",
    "Bounding boxes": "bounding_boxes",
    "Visible forgery traces": "visible_forgery_traces",
}

def normalize_keys(data):
    out = {}
    for k, v in data.items():
        out[ALIASES.get(k, k)] = v
    return out

def validate_one(name, data):
    errors = []
    data = normalize_keys(data)

    missing = CANONICAL_KEYS - set(data.keys())
    if missing:
        errors.append(f"missing keys: {sorted(missing)}")
        return errors

    cls = data["classification_result"]
    if not isinstance(cls, str) or cls.lower() not in {"real", "fake"}:
        errors.append("classification_result must be 'real' or 'fake'")

    boxes = data["bounding_boxes"]
    if cls.lower() == "real":
        if boxes not in ([], None, "None"):
            errors.append("real images should use empty boxes, None, or 'None'")
    else:
        if not isinstance(boxes, list):
            errors.append("fake images should use a list of bounding boxes")
        else:
            for i, box in enumerate(boxes):
                if not (isinstance(box, list) and len(box) == 4):
                    errors.append(f"box {i} must be [x1, y1, x2, y2]")
                    continue
                if not all(isinstance(x, (int, float)) for x in box):
                    errors.append(f"box {i} coordinates must be numbers")
                    continue
                if not all(1 <= float(x) <= 1000 for x in box):
                    errors.append(f"box {i} coordinates must be in [1, 1000]")
                if not (box[0] < box[2] and box[1] < box[3]):
                    errors.append(f"box {i} must satisfy x1 < x2 and y1 < y2")

    traces = data["visible_forgery_traces"]
    if not isinstance(traces, str) or len(traces.strip()) == 0:
        errors.append("visible_forgery_traces must be a non-empty string")

    return errors

def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/check_submission_format.py <submission.zip>")
        sys.exit(1)

    zip_path = Path(sys.argv[1])
    if not zip_path.exists():
        print(f"File not found: {zip_path}")
        sys.exit(1)

    total = 0
    bad = 0

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = [n for n in zf.namelist() if n.endswith(".json")]
        if not names:
            print("No JSON files found in zip.")
            sys.exit(1)

        for name in names:
            total += 1
            try:
                data = json.loads(zf.read(name).decode("utf-8"))
            except Exception as e:
                bad += 1
                print(f"[BAD] {name}: invalid JSON: {e}")
                continue

            errors = validate_one(name, data)
            if errors:
                bad += 1
                print(f"[BAD] {name}: {'; '.join(errors)}")

    print(f"Checked {total} JSON files. Invalid files: {bad}.")
    if bad:
        sys.exit(1)
    print("Submission format looks valid.")

if __name__ == "__main__":
    main()
