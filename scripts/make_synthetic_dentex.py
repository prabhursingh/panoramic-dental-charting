"""Make a tiny fake DENTEX download (same zip / json layout) for testing the code on a CPU.

    python scripts/make_synthetic_dentex.py --out /tmp/fake_dentex

writes training_data.zip, validation_data.zip and validation_triple.json. The "X-rays" are
simple drawings of teeth; any number obtained on them says nothing about real performance.
"""

import argparse
import io
import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

DIAGNOSES = ("Impacted", "Caries", "Periapical Lesion", "Deep Caries")
W, H, MID = 640, 300, 150


def tooth_box(fdi):
    """Box of a tooth on the fake OPG (patient's right on the viewer's left)."""
    q, t = divmod(fdi, 10)
    width = 18 + 2 * t
    offset = sum(18 + 2 * k for k in range(1, t)) + width / 2
    cx = W / 2 - offset if q in (1, 4) else W / 2 + offset
    if q in (1, 2):
        return [cx - width / 2, MID - 95, cx + width / 2, MID - 6]
    return [cx - width / 2, MID + 6, cx + width / 2, MID + 95]


def draw_xray(rng, teeth, findings):
    img = Image.fromarray((rng.normal(0.22, 0.03, (H, W)).clip(0, 1) * 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    d.rectangle([60, MID - 110, W - 60, MID + 110], fill=90)
    for fdi in teeth:
        x0, y0, x1, y1 = tooth_box(fdi)
        upper = fdi // 10 in (1, 2)
        crown = [x0, (y0 + y1) / 2, x1, y1] if upper else [x0, y0, x1, (y0 + y1) / 2]
        root = [x0 + 3, y0, x1 - 3, (y0 + y1) / 2] if upper else [x0 + 3, (y0 + y1) / 2, x1 - 3, y1]
        d.rectangle(root, fill=175)
        d.rectangle(crown, fill=225)
    for fdi, dx in findings:
        x0, y0, x1, y1 = tooth_box(fdi)
        upper = fdi // 10 in (1, 2)
        cx = (x0 + x1) / 2
        crown_y = y1 - 18 if upper else y0 + 18
        apex_y = y0 + 4 if upper else y1 - 4
        if dx == "Caries":
            d.ellipse([cx - 4, crown_y - 4, cx + 4, crown_y + 4], fill=70)
        elif dx == "Deep Caries":
            d.ellipse([cx - 7, crown_y - 12, cx + 7, crown_y + 12], fill=50)
        elif dx == "Periapical Lesion":
            d.ellipse([cx - 8, apex_y - 8, cx + 8, apex_y + 8], fill=45)
        elif dx == "Impacted":
            d.polygon([(x0, y0 + 20), (x1, y0 + 5), (x1, y1 - 5), (x0, y1)], fill=235)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def sample_case(rng, with_findings):
    teeth = [q * 10 + t for q in (1, 2, 3, 4) for t in range(1, 9)
             if rng.random() > (0.35 if t == 8 else 0.07)]
    findings = []
    if with_findings:
        for fdi in rng.choice(teeth, size=min(len(teeth), rng.integers(1, 5)), replace=False):
            fdi = int(fdi)
            dx = "Impacted" if fdi % 10 == 8 else str(rng.choice(DIAGNOSES[1:]))
            findings.append((fdi, dx))
    return teeth, findings


def annotation(ann_id, image_id, fdi, dx=None, level=3):
    x0, y0, x1, y1 = tooth_box(fdi)
    a = {"iscrowd": 0, "image_id": image_id, "bbox": [x0, y0, x1 - x0, y1 - y0],
         "segmentation": [[x0, y0, x1, y0, x1, y1, x0, y1]], "id": ann_id,
         "area": (x1 - x0) * (y1 - y0), "category_id_1": fdi // 10 - 1}
    if level >= 2:
        a["category_id_2"] = fdi % 10 - 1
    if level >= 3:
        a["category_id_3"] = DIAGNOSES.index(dx)
    return a


CATEGORIES = {"categories_1": [{"id": i, "name": str(i + 1), "supercategory": str(i + 1)} for i in range(4)],
              "categories_2": [{"id": i, "name": str(i + 1), "supercategory": str(i + 1)} for i in range(8)],
              "categories_3": [{"id": i, "name": n, "supercategory": n} for i, n in enumerate(DIAGNOSES)]}


def make_subset(z, rng, folder, json_name, n, level, prefix="train"):
    data = {"images": [], "annotations": []}
    ann_id = 1
    for k in range(n):
        teeth, findings = sample_case(rng, with_findings=level == 3)
        name = f"{prefix}_{k}.png"
        z.writestr(f"{folder}/xrays/{name}", draw_xray(rng, teeth, findings))
        data["images"].append({"height": H, "width": W, "id": k + 1, "file_name": name})
        items = [(f, dx) for f, dx in findings] if level == 3 else [(f, None) for f in teeth]
        for fdi, dx in items:
            data["annotations"].append(annotation(ann_id, k + 1, fdi, dx, level))
            ann_id += 1
    data.update({key: val for key, val in CATEGORIES.items() if int(key[-1]) <= level})
    z.writestr(f"{folder}/{json_name}", json.dumps(data))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=24, help="X-rays per labelled subset")
    ap.add_argument("--n-val", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    with zipfile.ZipFile(out / "training_data.zip", "w") as z:
        make_subset(z, rng, "training_data/quadrant", "train_quadrant.json", a.n, level=1)
        make_subset(z, rng, "training_data/quadrant_enumeration", "train_quadrant_enumeration.json", a.n, level=2)
        make_subset(z, rng, "training_data/quadrant_enumeration_disease",
                    "train_quadrant_enumeration_disease.json", a.n, level=3)
        for k in range(3):
            z.writestr(f"training_data/unlabelled/xrays/train_{k}.png", draw_xray(rng, [], []))
    with zipfile.ZipFile(out / "validation_data.zip", "w") as z:
        val = {"images": [], "annotations": []}
        ann_id = 1
        for k in range(a.n_val):
            teeth, findings = sample_case(rng, with_findings=True)
            name = f"val_{k}.png"
            z.writestr(f"validation_data/quadrant_enumeration_disease/xrays/{name}", draw_xray(rng, teeth, findings))
            val["images"].append({"height": H, "width": W, "id": k + 1, "file_name": name})
            for fdi, dx in findings:
                val["annotations"].append(annotation(ann_id, k + 1, fdi, dx, 3))
                ann_id += 1
    (out / "validation_triple.json").write_text(json.dumps(val))
    print(f"fake DENTEX written to {out}")


if __name__ == "__main__":
    main()
