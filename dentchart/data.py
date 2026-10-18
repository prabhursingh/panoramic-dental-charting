"""DENTEX data: unpacking the download, reading annotations, splits and the PyTorch dataset."""

import json
import random
import shutil
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .fdi import (DIAGNOSES, LABEL_TO_TOOTH, TOOTH_TO_LABEL, canonical_diagnosis, fdi_from_dentex,
                  mirror)

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


# --------------------------------------------------------------------------------------
# 1. Unpack the DENTEX download
# --------------------------------------------------------------------------------------

def _subset_of(json_path, annotations):
    """Which DENTEX level a json file belongs to: 'disease', 'enumeration' or 'quadrant'.

    The file path decides when it is clear (DENTEX folders are named after the level);
    otherwise the labels inside the file do.
    """
    low = json_path.lower()
    for name in ("disease", "enumeration", "quadrant"):
        if name in low:
            return name
    keys = set()
    for a in annotations[:50]:
        keys.update(a.keys())
    if "category_id_3" in keys:
        return "disease"
    if "category_id_2" in keys:
        return "enumeration"
    return "quadrant"


def _merge(base, extra):
    """Append one COCO-style dict to another, renumbering ids so they stay unique."""
    if base is None:
        return extra
    img_offset = max((im["id"] for im in base["images"]), default=0)
    ann_offset = max((a["id"] for a in base["annotations"]), default=0)
    for im in extra["images"]:
        im["id"] += img_offset
    for a in extra["annotations"]:
        a["id"] += ann_offset
        a["image_id"] += img_offset
    base["images"] += extra["images"]
    base["annotations"] += extra["annotations"]
    return base


def prepare_dentex(train_zip, out_dir, val_zip=None, val_json=None,
                   subsets=("enumeration", "disease"), log=print):
    """Extract the parts of DENTEX this project uses into a simple folder layout.

    out_dir/enumeration/   634 X-rays, every tooth boxed with its FDI number
    out_dir/disease/       705 X-rays, abnormal teeth boxed with FDI number + diagnosis
    out_dir/validation/    the 50 official validation X-rays (used for the dentist study)

    Each folder gets images/ and annotations.json. The folder names inside the zip are not
    assumed: every json file is opened and recognised by the labels it contains.
    """
    out = Path(out_dir)
    found = {}
    with zipfile.ZipFile(train_zip) as z:
        names = [n for n in z.namelist() if not n.startswith("__MACOSX")]
        for jn in [n for n in names if n.lower().endswith(".json")]:
            data = json.loads(z.read(jn))
            if not isinstance(data, dict) or "images" not in data or "annotations" not in data:
                continue
            subset = _subset_of(jn, data["annotations"])
            log(f"  found {jn}: {len(data['images'])} images -> '{subset}'")
            if subset not in subsets:
                continue
            root = jn.rsplit("/", 1)[0] + "/" if "/" in jn else ""
            index = {n.rsplit("/", 1)[-1]: n for n in names
                     if n.startswith(root) and n.lower().endswith(IMAGE_SUFFIXES)}
            img_dir = out / subset / "images"
            img_dir.mkdir(parents=True, exist_ok=True)
            missing = 0
            for im in data["images"]:
                name = Path(im["file_name"]).name
                src = index.get(name)
                if src is None:
                    missing += 1
                    continue
                im["file_name"] = name
                dst = img_dir / name
                if not dst.exists():
                    with z.open(src) as fsrc, open(dst, "wb") as fdst:
                        shutil.copyfileobj(fsrc, fdst)
            if missing:
                log(f"  warning: {missing} images listed in {jn} were not found in the zip")
            prev = found.get(subset, {}).get("data")
            found[subset] = {"data": _merge(prev, data), "sources": found.get(subset, {}).get("sources", []) + [jn]}

    summary = {}
    for subset, item in found.items():
        (out / subset / "annotations.json").write_text(json.dumps(item["data"]))
        summary[subset] = {"images": len(item["data"]["images"]),
                           "annotations": len(item["data"]["annotations"]),
                           "source": item["sources"]}

    if val_zip and val_json:
        data = json.loads(Path(val_json).read_text())
        wanted = {Path(im["file_name"]).name for im in data["images"]}
        img_dir = out / "validation" / "images"
        img_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(val_zip) as z:
            for n in z.namelist():
                name = n.rsplit("/", 1)[-1]
                if name in wanted and not (img_dir / name).exists():
                    with z.open(n) as fsrc, open(img_dir / name, "wb") as fdst:
                        shutil.copyfileobj(fsrc, fdst)
        for im in data["images"]:
            im["file_name"] = Path(im["file_name"]).name
        (out / "validation" / "annotations.json").write_text(json.dumps(data))
        summary["validation"] = {"images": len(data["images"]), "annotations": len(data["annotations"]),
                                 "images_found": len(list(img_dir.iterdir()))}

    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


# --------------------------------------------------------------------------------------
# 2. Read annotations into simple records
# --------------------------------------------------------------------------------------

def diagnosis_names(data):
    """Diagnosis names in category_id_3 order, read from the file when it lists them."""
    cats = data.get("categories_3")
    if cats:
        return tuple(canonical_diagnosis(c["name"]) for c in sorted(cats, key=lambda c: c["id"]))
    return DIAGNOSES


def load_records(subset_dir, diagnoses=None):
    """Read one subset folder into a list of records (one per X-ray).

    Each record: {file_name, width, height, anns: [{box, fdi, diagnosis, polygon}]}
    with box = [x0, y0, x1, y1] in pixels. Returns (records, diagnosis_names).
    """
    subset_dir = Path(subset_dir)
    data = json.loads((subset_dir / "annotations.json").read_text())
    names = tuple(diagnoses) if diagnoses else diagnosis_names(data)
    records = {}
    for im in data["images"]:
        records[im["id"]] = {"file_name": Path(im["file_name"]).name, "width": im["width"],
                             "height": im["height"], "anns": []}
    for a in data["annotations"]:
        rec = records.get(a["image_id"])
        x, y, w, h = a["bbox"]
        if rec is None or w < 1 or h < 1:
            continue
        fdi = fdi_from_dentex(a["category_id_1"], a["category_id_2"]) if "category_id_2" in a else None
        dx = names[a["category_id_3"]] if "category_id_3" in a else None
        seg = a.get("segmentation")
        polygon = seg[0] if isinstance(seg, list) and seg and isinstance(seg[0], list) else None
        rec["anns"].append({"box": [x, y, x + w, y + h], "fdi": fdi, "diagnosis": dx, "polygon": polygon})
    image_dir = subset_dir / "images"
    out = [r for r in records.values() if (image_dir / r["file_name"]).exists()]
    out.sort(key=lambda r: r["file_name"])
    return out, names


def split_records(records, val=0.10, test=0.15, seed=42):
    """Reproducible train / validation / test split at the level of X-rays (patients)."""
    shuffled = sorted(records, key=lambda r: r["file_name"])
    random.Random(seed).shuffle(shuffled)
    n_test, n_val = round(len(shuffled) * test), round(len(shuffled) * val)
    return shuffled[n_test + n_val:], shuffled[n_test:n_test + n_val], shuffled[:n_test]


# --------------------------------------------------------------------------------------
# 3. PyTorch dataset
# --------------------------------------------------------------------------------------

def read_xray(path):
    """Read an X-ray as a float32 array (H, W) in [0, 1]; works for 8-bit and 16-bit images."""
    img = Image.open(path)
    if img.mode.startswith("I"):  # 16-bit or 32-bit integer grayscale
        arr = np.asarray(img, dtype=np.float32)
        return arr / max(float(arr.max()), 1.0)
    return np.asarray(img.convert("L"), dtype=np.float32) / 255.0


class OPGDataset(torch.utils.data.Dataset):
    """Panoramic X-rays with boxes for one of two tasks.

    task="teeth":    one class per FDI tooth (32 classes) -> the dental chart
    task="findings": one class per diagnosis (4 classes) -> caries, deep caries, ...

    Training augmentation flips the X-ray left-right. A flip moves teeth to the other side
    of the mouth, so the FDI numbers are mirrored too (36 becomes 46): flipping without
    relabelling would teach the model the wrong side.
    """

    def __init__(self, records, image_dir, task, diagnoses=DIAGNOSES, train=False,
                 flip_p=0.5, jitter=0.15):
        assert task in ("teeth", "findings")
        self.records, self.image_dir, self.task = list(records), Path(image_dir), task
        self.diagnoses, self.train = tuple(diagnoses), train
        self.flip_p, self.jitter = flip_p, jitter

    @property
    def class_names(self):
        if self.task == "teeth":
            return [str(LABEL_TO_TOOTH[i]) for i in range(1, len(LABEL_TO_TOOTH) + 1)]
        return list(self.diagnoses)

    @property
    def num_classes(self):
        return len(self.class_names) + 1  # + background

    def __len__(self):
        return len(self.records)

    def image(self, i):
        """Grayscale X-ray as float tensor (H, W) in [0, 1]."""
        return torch.from_numpy(read_xray(self.image_dir / self.records[i]["file_name"]))

    def annotations(self, i):
        """Boxes, class labels and FDI numbers of X-ray i (no augmentation)."""
        boxes, labels, fdis = [], [], []
        for a in self.records[i]["anns"]:
            if self.task == "teeth":
                if a["fdi"] not in TOOTH_TO_LABEL:
                    continue
                label = TOOTH_TO_LABEL[a["fdi"]]
            else:
                if a["diagnosis"] not in self.diagnoses:
                    continue
                label = self.diagnoses.index(a["diagnosis"]) + 1
            boxes.append(a["box"])
            labels.append(label)
            fdis.append(a["fdi"] or 0)
        return (torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
                torch.tensor(labels, dtype=torch.int64), fdis)

    def __getitem__(self, i):
        x = self.image(i)
        boxes, labels, fdis = self.annotations(i)
        if self.train:
            if random.random() < self.flip_p:
                width = x.shape[1]
                x = x.flip(-1)
                boxes = boxes[:, [2, 1, 0, 3]]
                boxes[:, [0, 2]] = width - boxes[:, [0, 2]]
                fdis = [mirror(f) if f else 0 for f in fdis]
                if self.task == "teeth":
                    labels = torch.tensor([TOOTH_TO_LABEL[f] for f in fdis], dtype=torch.int64)
            if self.jitter:
                contrast = 1 + random.uniform(-self.jitter, self.jitter)
                brightness = random.uniform(-self.jitter, self.jitter) / 2
                mean = x.mean()
                x = ((x - mean) * contrast + mean + brightness).clamp(0, 1)
        image = x.unsqueeze(0).expand(3, -1, -1).contiguous()
        return image, {"boxes": boxes, "labels": labels}


def collate(batch):
    return tuple(zip(*batch))
