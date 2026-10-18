"""Chart any panoramic X-ray with the trained models (used by the dashboard, app/dashboard.py).

    from dentchart import infer
    teeth_model, find_model, device, thresh, info = infer.load_models([WORK])
    result = infer.chart_opg("my_opg.jpg", teeth_model, find_model, find_thresh=thresh, device=device)
    result["findings"]      # [{"fdi": 36, "diagnosis": "Deep Caries", "score": 0.93, "box": [...]}, ...]
    infer.plot_result(result)

Made for X-rays the models have not seen, also from other machines, crops and screenshots:
  * the trained weights are loaded strictly: a missing or mismatched checkpoint is an error,
    never a silently untrained model (which gives convincing-looking nonsense);
  * the X-ray is prepared like the training X-rays: one grey channel, 0-1 (8-bit, 16-bit or RGB);
  * dark or flat strips at the edges (a scanner's label band with its name and an "R", white
    frames) are cropped;
  * scale fit: a crop, a screenshot or another machine can make teeth look bigger, smaller or
    squeezed. Unless the X-ray has the usual OPG shape and the tooth model finds most teeth, it
    is also tried smaller, stretched to the usual shape and with stretched contrast; the setting
    where the tooth model finds the most teeth is used and reported;
  * a finding is charted only if it lies on a detected tooth (one entry per tooth and diagnosis).
Research use only - not a medical device and not a diagnosis.
"""

import json
import os
import textwrap
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

from .evaluate import assign_teeth
from .fdi import CODES, COLORS, DIAGNOSES, LABEL_TO_TOOTH, is_upper
from .model import build_model

MAX_SIZE = {"teeth": 1333, "findings": 1600}  # the notebook's training settings
TARGET_ASPECT = 2.1      # width / height of a typical DENTEX OPG
ZOOMS = (1.0, 0.75, 0.5)
ENOUGH_TEETH = 20        # an OPG-shaped X-ray with this many teeth is used as it is


# --------------------------------------------------------------------------------------
# 1. Prepare the X-ray like the training data
# --------------------------------------------------------------------------------------

def to_grey(image):
    """File, PIL image, array or tensor -> float32 array (H, W) in [0, 1].

    Same scaling as the training X-rays (dentchart.data.read_xray): 8-bit / 255, 16-bit / max.
    """
    if isinstance(image, (str, os.PathLike)):
        image = Image.open(image)
    if isinstance(image, Image.Image):
        image = ImageOps.exif_transpose(image)
        if image.mode.startswith("I"):  # 16-bit (or 32-bit integer) grey
            arr = np.asarray(image, dtype=np.float32)
            return np.clip(arr / max(float(arr.max()), 1.0), 0, 1)
        if image.mode != "F":
            return np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    if torch.is_tensor(image):
        image = image.detach().cpu().float().numpy()
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim == 3:  # (C, H, W) or (H, W, C)
        channels_first = arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4)
        arr = arr[:3].mean(axis=0) if channels_first else arr[..., :3].mean(axis=-1)
    if arr.max() > 1.0:
        arr = arr / (255.0 if arr.max() <= 255 else float(arr.max()))
    return np.clip(arr, 0, 1).astype(np.float32)


def _edge_strips(level, std, dark, flat):
    """Ends of the strip runs at both sides of a column (or row) profile."""
    strip = (level < dark * np.median(level)) | (std < flat)
    lo, hi = 0, len(strip)
    while lo < hi and strip[lo]:
        lo += 1
    while hi > lo and strip[hi - 1]:
        hi -= 1
    return lo, hi


def find_xray_area(arr, dark=0.35, flat=0.01, max_cut=0.25):
    """Region (x0, y0, x1, y1) of the X-ray inside dark or flat edge strips.

    An edge column (row) belongs to a strip when its median brightness is below `dark` x that of
    a typical column (row) - a black label band, even with the machine name and a big "R" on it -
    or when it is flat (std < `flat`), like a white frame. Only strips touching the edge count,
    and at most `max_cut` of the width or height is cut on each side.
    """
    h, w = arr.shape
    x0, y0, x1, y1 = 0, 0, w, h
    for _ in range(2):  # strips can hide behind each other (white frame, then a black band)
        sub = arr[y0:y1, x0:x1]
        lo, hi = _edge_strips(np.median(sub, 0), sub.std(0), dark, flat)
        x0, x1 = min(x0 + lo, int(max_cut * w)), max(x0 + hi, w - int(max_cut * w))
        sub = arr[y0:y1, x0:x1]
        lo, hi = _edge_strips(np.median(sub, 1), sub.std(1), dark, flat)
        y0, y1 = min(y0 + lo, int(max_cut * h)), max(y0 + hi, h - int(max_cut * h))
    return x0, y0, x1, y1


def stretch_contrast(arr, lo=0.5, hi=99.5):
    """Map the lo-hi percentiles of the grey levels to 0-1 (for flat, low-contrast exports)."""
    a, b = np.percentile(arr, [lo, hi])
    return np.clip((arr - a) / max(float(b - a), 1e-6), 0, 1).astype(np.float32)


def _resize_width(arr, factor):
    if abs(factor - 1) < 1e-3:
        return arr
    h, w = arr.shape
    t = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))[None, None]
    t = F.interpolate(t, size=(h, max(1, round(w * factor))), mode="bilinear", align_corners=False)
    return t[0, 0].numpy()


# --------------------------------------------------------------------------------------
# 2. Run the models at the scale they were trained at
# --------------------------------------------------------------------------------------

def own_scale(model, h, w):
    """The resize factor a torchvision detector applies to an (h, w) image by itself."""
    return min(model.transform.min_size[-1] / min(h, w), model.transform.max_size / max(h, w))


@torch.no_grad()
def detect(model, arr, scale=None, device="cpu"):
    """Run a detector on a grey X-ray (H, W) in [0, 1]; boxes come back in arr's pixels.

    scale=None lets the model resize the X-ray itself, exactly as in training; a number makes
    the model see the X-ray resized by exactly that factor.
    """
    tr = model.transform
    old = (tr.min_size, tr.max_size)
    if scale is not None:
        h, w = arr.shape
        tr.min_size, tr.max_size = (scale * min(h, w),), scale * max(h, w)
    try:
        x = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
        out = model([x[None].expand(3, -1, -1).contiguous().to(device)])[0]
    finally:
        tr.min_size, tr.max_size = old
    return {"boxes": out["boxes"].float().cpu().numpy().reshape(-1, 4),
            "scores": out["scores"].float().cpu().numpy(),
            "labels": out["labels"].cpu().numpy().astype(np.int64)}


def fit_scale(arr, teeth_model, tooth_thresh=0.5, device="cpu", search=True):
    """Find the setting at which the tooth model sees the most teeth.

    The X-ray as it is (exactly the training setting) is tried first and kept if it has the usual
    OPG shape and the tooth model finds at least ENOUGH_TEETH teeth. Otherwise smaller zooms, a
    horizontal stretch to the usual OPG shape (for squeezed images) and a contrast stretch are
    tried as well: more teeth wins, then a clearly higher mean tooth score.
    Returns (setting, prepared X-ray, tooth prediction, list of everything tried); setting["scale"]
    is the tooth model's resize factor (None = its own, as in training).
    """
    h, w = arr.shape
    stretches = [1.0] + ([TARGET_ASPECT * h / w] if w / h < 1.6 else [])
    settings = [(s, z, c) for c in (False, True) for s in stretches for z in ZOOMS]
    best, tried = None, []
    for s, z, c in settings:
        img = _resize_width(stretch_contrast(arr) if c else arr, s)
        scale = None if (s, z, c) == settings[0] else z * own_scale(teeth_model, *img.shape)
        pred = detect(teeth_model, img, scale, device)
        keep = pred["scores"] >= tooth_thresh
        n = len(set(pred["labels"][keep].tolist()))
        mean = float(pred["scores"][keep].mean()) if keep.any() else 0.0
        setting = {"stretch": img.shape[1] / w, "zoom": z, "contrast": c, "scale": scale}
        tried.append({**setting, "teeth": n, "mean_score": round(mean, 3)})
        if best is None or n > best[0] or (n == best[0] and mean > best[1] + 0.02):
            best = (n, mean, setting, img, pred)
        if not search or (len(tried) == 1 and n >= ENOUGH_TEETH and len(stretches) == 1):
            break
    return best[2], best[3], best[4], tried


def chart_opg(image, teeth_model, find_model, find_thresh=0.5, tooth_thresh=0.5, diagnoses=DIAGNOSES,
              device="cpu", fit=True, crop=True):
    """Chart one X-ray. All boxes in the result are in the pixels of the original image."""
    full = to_grey(image)
    H, W = full.shape
    x0, y0, x1, y1 = find_xray_area(full) if crop else (0, 0, W, H)
    arr = full[y0:y1, x0:x1]
    setting, img, tp, tried = fit_scale(arr, teeth_model, tooth_thresh, device, search=fit)
    # the findings model was trained on larger X-rays than the tooth model (1600 vs 1333 px wide):
    # keep that ratio when the scale is set by hand
    f_scale = None if setting["scale"] is None else (
        setting["scale"] * find_model.transform.max_size / teeth_model.transform.max_size)
    fp = detect(find_model, img, f_scale, device)
    keep = fp["scores"] >= find_thresh
    f_boxes, f_scores, f_labels = fp["boxes"][keep], fp["scores"][keep], fp["labels"][keep]
    teeth_of = assign_teeth(f_boxes, tp, tooth_thresh)

    def to_original(boxes):
        b = np.asarray(boxes, dtype=np.float32).reshape(-1, 4).copy()
        b[:, [0, 2]] = b[:, [0, 2]] / setting["stretch"] + x0
        b[:, [1, 3]] += y0
        return b

    tk = tp["scores"] >= tooth_thresh
    teeth = {"boxes": to_original(tp["boxes"][tk]), "scores": tp["scores"][tk],
             "fdi": [LABEL_TO_TOOTH[int(l)] for l in tp["labels"][tk]]}
    found = [{"fdi": int(t), "diagnosis": diagnoses[int(l) - 1], "score": float(s), "box": b}
             for t, l, s, b in zip(teeth_of, f_labels, f_scores, to_original(f_boxes))]
    best = {}  # one entry per tooth and diagnosis: the most confident box
    for f in sorted(found, key=lambda f: -f["score"]):
        if f["fdi"]:
            best.setdefault((f["fdi"], f["diagnosis"]), f)
    charted = sorted(best.values(), key=lambda f: (f["fdi"], -f["score"]))
    present = set(teeth["fdi"])

    notes, warnings = [], []
    if (x0, y0, x1, y1) != (0, 0, W, H):
        cut = [f"{n} {v} px" for n, v in (("left", x0), ("right", W - x1), ("top", y0), ("bottom", H - y1)) if v]
        notes.append("Edge strip cropped (scanner label / border): " + ", ".join(cut) + ".")
    changed = [f"stretched x{setting['stretch']:.2f} sideways" if abs(setting["stretch"] - 1) > 1e-3 else "",
               f"zoomed out to {setting['zoom']:.0%}" if setting["zoom"] != 1 else "",
               "contrast stretched" if setting["contrast"] else ""]
    changed = [c for c in changed if c]
    if changed:
        notes.append("Scale fit: " + ", ".join(changed) + " to match the training X-rays.")
        warnings.append("This X-ray differs from the training X-rays in size, shape or contrast, so "
                        "results are less reliable than the test-set results.")
    if len(present) < 10:
        warnings.append(f"Only {len(present)} teeth found. Unless the patient really has few teeth, this "
                        "X-ray is too different from the training X-rays (machine, crop or resolution): "
                        "do not trust these findings.")
    return {"image": full, "crop": (x0, y0, x1, y1), "setting": setting, "tried": tried,
            "teeth": teeth, "present": present, "findings": charted,
            "unassigned": [f for f in found if not f["fdi"]],
            "chart": {(f["fdi"], f["diagnosis"]) for f in charted},
            "find_thresh": find_thresh, "notes": notes, "warnings": warnings}


# --------------------------------------------------------------------------------------
# 3. Report and figure
# --------------------------------------------------------------------------------------

def report_lines(result):
    lines = [f"Teeth found: {len(result['present'])} of 32",
             f"Findings on the chart: {len(result['findings'])} (score >= {result['find_thresh']:.2f})"]
    if result["unassigned"]:
        lines.append(f"Not on a detected tooth (not charted): {len(result['unassigned'])}")
    lines.append("")
    for f in result["findings"]:
        lines.append(f"  {f['fdi']}  {f['diagnosis']:<18} {CODES.get(f['diagnosis'], ''):<3} score {f['score']:.2f}")
    if not result["findings"]:
        lines.append("  No findings above the threshold.")
    return lines


def plot_result(result, title=None, show_teeth=True, show_unassigned=False, report=True):
    """X-ray with tooth numbers and findings, the dental chart and (report=True) the list of findings."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    from .viz import plot_dental_chart

    full = result["image"]
    H, W = full.shape
    xray_h = min(9.0, 15 * H / W)
    fig = plt.figure(figsize=(15, xray_h + 3.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[xray_h, 3.4], width_ratios=[0.6, 0.4] if report else [0.8, 0.2],
                          hspace=0.08, wspace=0.04)
    ax = fig.add_subplot(gs[0, :])
    ax.imshow(full, cmap="gray", vmin=0, vmax=1)
    x0, y0, x1, y1 = result["crop"]
    if (x0, y0, x1, y1) != (0, 0, W, H):
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="#ffd166", lw=1, ls="--"))
    font = 7
    if show_teeth:
        for b, t in zip(result["teeth"]["boxes"], result["teeth"]["fdi"]):
            ax.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec="#8ecae6", lw=0.7))
            upper = is_upper(t)
            ax.text((b[0] + b[2]) / 2, b[1] if upper else b[3], str(t), color="#8ecae6", fontsize=font,
                    ha="center", va="bottom" if upper else "top")
    draw = [(f, COLORS.get(f["diagnosis"], "#888888"), "-") for f in result["findings"]]
    if show_unassigned:
        draw += [(f, "#9e9e9e", "--") for f in result["unassigned"]]
    for f, color, style in draw:
        b = f["box"]
        ax.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec=color, lw=1.8, ls=style))
        ax.text(b[0] + 2, b[1] + 2, f"{f['fdi'] or '?'} {CODES.get(f['diagnosis'], f['diagnosis'])}",
                color="white", fontsize=font, va="top", ha="left",
                bbox=dict(facecolor=color, alpha=0.85, pad=1, lw=0))
    ax.set_axis_off()
    ax.set_title(title or "AI dental chart (research use only, not a diagnosis)", fontsize=11)

    ax_chart = fig.add_subplot(gs[1, 0] if report else gs[1, :])
    plot_dental_chart(result["chart"], present=result["present"], ax=ax_chart,
                      title="Dental chart from the AI (grey = tooth not found)")
    if not report:
        return fig
    ax_text = fig.add_subplot(gs[1, 1])
    ax_text.set_axis_off()
    lines = report_lines(result)
    if len(lines) > 12:
        lines = lines[:11] + [f"  ... and {len(lines) - 11} more (see the printed list)"]
    blocks = [("\n".join(lines), "#222222", "monospace", 8.5)]
    blocks += [(textwrap.fill(n, 64), "#555555", "sans-serif", 8) for n in result["notes"]]
    blocks += [(textwrap.fill(w, 64), "#c0392b", "sans-serif", 8) for w in result["warnings"]]
    height_in = ax_text.get_position().height * fig.get_figheight()
    y = 1.0
    for text, color, family, size in blocks:
        ax_text.text(0, y, text, va="top", ha="left", color=color, family=family, fontsize=size,
                     transform=ax_text.transAxes)
        y -= (text.count("\n") + 1) * size * 1.35 / 72 / height_in + 0.04
    return fig


# --------------------------------------------------------------------------------------
# 4. Trained models, and a check on DENTEX
# --------------------------------------------------------------------------------------

def find_work(work_dirs):
    """The first folder that holds both trained models (runs/teeth/best.pt, runs/findings/best.pt)."""
    for d in map(Path, work_dirs):
        if all((d / "runs" / name / "best.pt").exists() for name in ("teeth", "findings")):
            return d
    raise FileNotFoundError("trained models not found (runs/teeth/best.pt and runs/findings/best.pt) in "
                            + ", ".join(str(d) for d in work_dirs)
                            + ". Run Parts 1 and 2 of the notebook first.")


def saved_threshold(work, default=0.5):
    """The finding threshold chosen on the validation X-rays (WORK/results.json), else `default`."""
    try:
        return float(json.loads((Path(work) / "results.json").read_text())["findings"]["threshold"])
    except (OSError, KeyError, TypeError, ValueError):
        return default


def load_models(work_dirs, device=None):
    """Load both trained models strictly from the first folder in `work_dirs` that has them.

    Returns (teeth_model, findings_model, device, finding threshold, description).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    work = find_work(work_dirs)
    expected = {"teeth": len(LABEL_TO_TOOTH) + 1, "findings": len(DIAGNOSES) + 1}   # + background
    models, info = [], []
    for name in ("teeth", "findings"):
        path = work / "runs" / name / "best.pt"
        ck = torch.load(path, map_location=device, weights_only=False)
        state = ck["model"] if "model" in ck else ck   # best.pt holds {"model": weights, "epoch", "val"}
        num_classes = state["roi_heads.box_predictor.cls_score.weight"].shape[0]
        if num_classes != expected[name]:
            raise ValueError(f"{path} predicts {num_classes - 1} classes, the {name} model needs "
                             f"{expected[name] - 1}: are the two checkpoints swapped?")
        model = build_model(num_classes, pretrained=False, max_size=MAX_SIZE[name])
        model.load_state_dict(state)  # strict: fails loudly instead of keeping untrained layers
        model.to(device).eval()
        models.append(model)
        ap50 = (ck.get("val") or {}).get("AP50")
        info.append(f"{name} model from epoch {ck.get('epoch', '?')}"
                    + (f" (validation AP50 {ap50:.2f})" if ap50 is not None else ""))
    return models[0], models[1], device, saved_threshold(work), ", ".join(info) + f" - {work}"


def _codes(findings):
    return ", ".join(f"{t} {CODES.get(d, d)}" for t, d in sorted(findings)) or "none"


def check_on_dentex(data_dir, work_dirs, k=0, show=True):
    """Chart a DENTEX test X-ray (never used in training) and a copy made to look like a screenshot
    from another system (black label band added, squeezed sideways), and compare both with the labels."""
    import matplotlib.pyplot as plt
    from PIL import ImageDraw

    from . import data as D

    subset = Path(data_dir) / "disease"
    if not (subset / "annotations.json").exists():
        print(f"DENTEX is not on this runtime ({subset}): run Part 0 first to do this check.")
        return []
    recs, dxs = D.load_records(subset)
    _, _, test = D.split_records(recs, seed=42)   # the notebook's test split
    rec = sorted(test, key=lambda r: (-len(r["anns"]), r["file_name"]))[k]
    teeth_model, find_model, device, thresh, info = load_models(work_dirs)
    print("Loaded:", info, f"| finding threshold {thresh:.2f}")
    truth = {(a["fdi"], a["diagnosis"]) for a in rec["anns"] if a["fdi"] and a["diagnosis"] in dxs}
    original = to_grey(subset / "images" / rec["file_name"])
    h, w = original.shape
    band = Image.new("L", (max(w // 20, 30), h), 0)
    ImageDraw.Draw(band).text((10, h // 2), "SCANNER", fill=255)
    ImageDraw.Draw(band).rectangle([10, h - 120, band.width - 10, h - 40], fill=255)   # an "R" marker
    screenshot = np.concatenate([np.asarray(band, dtype=np.float32) / 255, original], axis=1)
    screenshot = np.asarray(Image.fromarray(screenshot).resize((int(screenshot.shape[1] * 0.45), h),
                                                               Image.BILINEAR))
    results = []
    for name, img in (("DENTEX test X-ray", original), ("same X-ray with a label band, squeezed", screenshot)):
        r = chart_opg(img, teeth_model, find_model, find_thresh=thresh, diagnoses=dxs, device=device)
        hit = r["chart"] & truth
        print(f"\n{name} ({rec['file_name']}): {len(r['present'])} teeth found")
        print("  AI chart:     ", _codes(r["chart"]))
        print("  DENTEX labels:", _codes(truth), f"-> {len(hit)} of {len(truth)} with tooth and diagnosis right")
        for line in r["notes"] + ["WARNING: " + x for x in r["warnings"]]:
            print("  " + line)
        if show:
            plot_result(r, title=name)
            plt.show()
        results.append(r)
    return results
