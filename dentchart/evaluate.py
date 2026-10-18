"""Evaluation: COCO detection metrics and clinically oriented dental-chart metrics."""

import contextlib
import io

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.ops import box_iou

from .data import collate
from .fdi import LABEL_TO_TOOTH, TEETH


# --------------------------------------------------------------------------------------
# Predictions
# --------------------------------------------------------------------------------------

@torch.no_grad()
def predict(model, ds, device=None, batch_size=2, num_workers=2):
    """Run the model on every X-ray of a dataset. Returns one dict per X-ray (numpy arrays)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    was_train, ds.train = ds.train, False  # never augment during evaluation
    out = []
    try:
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                            collate_fn=collate)
        for images, _ in loader:
            for p in model([im.to(device) for im in images]):
                out.append({k: p[k].detach().float().cpu().numpy() if k != "labels" else p[k].cpu().numpy()
                            for k in ("boxes", "scores", "labels")})
    finally:
        ds.train = was_train
    return out


# --------------------------------------------------------------------------------------
# COCO metrics (the standard object-detection scores used by DENTEX)
# --------------------------------------------------------------------------------------

def coco_metrics(preds, ds):
    """AP (IoU 0.50:0.95), AP50, AP75, AR100 and AP50 per class, computed with pycocotools."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    names = ds.class_names
    gt = {"images": [], "annotations": [], "categories": [{"id": i + 1, "name": n} for i, n in enumerate(names)]}
    dets, ann_id = [], 1
    for i, p in enumerate(preds):
        rec = ds.records[i]
        gt["images"].append({"id": i, "width": rec["width"], "height": rec["height"]})
        boxes, labels, _ = ds.annotations(i)
        for b, l in zip(boxes.tolist(), labels.tolist()):
            w, h = b[2] - b[0], b[3] - b[1]
            gt["annotations"].append({"id": ann_id, "image_id": i, "category_id": int(l),
                                      "bbox": [b[0], b[1], w, h], "area": w * h, "iscrowd": 0})
            ann_id += 1
        for b, s, l in zip(p["boxes"].tolist(), p["scores"].tolist(), p["labels"].tolist()):
            dets.append({"image_id": i, "category_id": int(l), "bbox": [b[0], b[1], b[2] - b[0], b[3] - b[1]],
                         "score": float(s)})
    empty = {"AP": 0.0, "AP50": 0.0, "AP75": 0.0, "AR100": 0.0, "AP50_per_class": {n: 0.0 for n in names}}
    if not dets or not gt["annotations"]:
        return empty
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO()
        coco_gt.dataset = gt
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(dets)
        ev = COCOeval(coco_gt, coco_dt, "bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    per_class = {}
    precision = ev.eval["precision"]  # [iou, recall, class, area, maxDets]
    for k, n in enumerate(names):
        pr = precision[0, :, k, 0, -1]
        per_class[n] = float(pr[pr > -1].mean()) if (pr > -1).any() else float("nan")
    s = [max(float(v), 0.0) for v in ev.stats]
    return {"AP": s[0], "AP50": s[1], "AP75": s[2], "AR100": s[8], "AP50_per_class": per_class}


def evaluate_map(model, ds, device=None, num_workers=2):
    """Shortcut used during training: predictions + COCO metrics (without the per-class dict)."""
    scores = coco_metrics(predict(model, ds, device=device, num_workers=num_workers), ds)
    return {k: v for k, v in scores.items() if k != "AP50_per_class"}


# --------------------------------------------------------------------------------------
# Dental chart from the tooth model
# --------------------------------------------------------------------------------------

def teeth_present(pred, score_thresh=0.5):
    """Set of FDI numbers the tooth model sees in one X-ray."""
    keep = pred["scores"] >= score_thresh
    return {LABEL_TO_TOOTH[int(l)] for l in pred["labels"][keep]}


def teeth_chart_metrics(preds, ds, score_thresh=0.5, iou_thresh=0.5):
    """How well the model charts present and missing teeth.

    chart_accuracy       share of the 32 tooth positions charted correctly (present / missing)
    missing_sensitivity  share of truly missing teeth the model calls missing
    missing_precision    share of teeth the model calls missing that are truly missing
    detection_rate       share of teeth found with a box (IoU >= 0.5, any number)
    numbering_accuracy   share of found teeth that got the correct FDI number
    """
    correct = total = miss_tp = miss_fn = miss_fp = found = right_number = n_gt = 0
    for i, p in enumerate(preds):
        boxes, labels, _ = ds.annotations(i)
        gt_present = {LABEL_TO_TOOTH[int(l)] for l in labels}
        pr_present = teeth_present(p, score_thresh)
        for fdi in TEETH:
            g, q = fdi in gt_present, fdi in pr_present
            correct += g == q
            total += 1
            miss_tp += (not g) and (not q)
            miss_fn += (not g) and q
            miss_fp += g and (not q)
        keep = p["scores"] >= score_thresh
        n_gt += len(boxes)
        if len(boxes) and keep.any():
            iou = box_iou(boxes, torch.as_tensor(p["boxes"][keep]))
            best_iou, best_j = iou.max(dim=1)
            pred_labels = p["labels"][keep]
            for g_label, bi, bj in zip(labels.tolist(), best_iou.tolist(), best_j.tolist()):
                if bi >= iou_thresh:
                    found += 1
                    right_number += int(pred_labels[bj]) == g_label
    return {"chart_accuracy": correct / max(total, 1),
            "missing_sensitivity": miss_tp / max(miss_tp + miss_fn, 1),
            "missing_precision": miss_tp / max(miss_tp + miss_fp, 1),
            "detection_rate": found / max(n_gt, 1),
            "numbering_accuracy": right_number / max(found, 1),
            "truly_missing_teeth": miss_tp + miss_fn}


# --------------------------------------------------------------------------------------
# Findings on the chart: "36 - deep caries"
# --------------------------------------------------------------------------------------

def assign_teeth(finding_boxes, teeth_pred, tooth_thresh=0.5, min_iou=0.3):
    """Give each finding box the FDI number of the tooth box it overlaps most (0 = none)."""
    keep = teeth_pred["scores"] >= tooth_thresh
    if len(finding_boxes) == 0 or not keep.any():
        return [0] * len(finding_boxes)
    iou = box_iou(torch.as_tensor(np.asarray(finding_boxes, dtype=np.float32)),
                  torch.as_tensor(teeth_pred["boxes"][keep]))
    best_iou, best_j = iou.max(dim=1)
    labels = teeth_pred["labels"][keep]
    return [LABEL_TO_TOOTH[int(labels[j])] if v >= min_iou else 0
            for v, j in zip(best_iou.tolist(), best_j.tolist())]


def chart_findings(find_pred, teeth_pred, diagnoses, find_thresh=0.5, tooth_thresh=0.5):
    """Set of (FDI, diagnosis) pairs for one X-ray, e.g. {(36, 'Deep Caries'), (48, 'Impacted')}."""
    keep = find_pred["scores"] >= find_thresh
    boxes = find_pred["boxes"][keep]
    teeth = assign_teeth(boxes, teeth_pred, tooth_thresh)
    return {(fdi, diagnoses[int(l) - 1]) for fdi, l in zip(teeth, find_pred["labels"][keep])}


def gt_findings(record, diagnoses):
    """Ground-truth (FDI, diagnosis) pairs of one record."""
    return {(a["fdi"], a["diagnosis"]) for a in record["anns"] if a["fdi"] and a["diagnosis"] in diagnoses}


def chart_scores(pred_sets, gt_sets, diagnoses):
    """Precision, recall (sensitivity) and F1 of charted findings, overall and per diagnosis.

    A finding counts as correct only if both the tooth number and the diagnosis are right.
    """
    rows = {}
    for dx in list(diagnoses) + ["All"]:
        tp = fp = fn = 0
        for p, g in zip(pred_sets, gt_sets):
            if dx != "All":
                p = {x for x in p if x[1] == dx}
                g = {x for x in g if x[1] == dx}
            tp += len(p & g)
            fp += len(p - g)
            fn += len(g - p)
        precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        rows[dx] = {"TP": tp, "FP": fp, "FN": fn, "precision": precision, "recall": recall,
                    "F1": 2 * precision * recall / max(precision + recall, 1e-9)}
    return rows


def best_threshold(find_preds, teeth_preds, gt_sets, diagnoses, grid=None):
    """Score threshold for findings that maximises chart-level F1 (chosen on validation data)."""
    grid = np.round(np.arange(0.2, 0.95, 0.05), 2) if grid is None else grid
    best = (0.5, -1.0)
    for t in grid:
        sets = [chart_findings(f, tp, diagnoses, find_thresh=t) for f, tp in zip(find_preds, teeth_preds)]
        f1 = chart_scores(sets, gt_sets, diagnoses)["All"]["F1"]
        if f1 > best[1]:
            best = (float(t), f1)
    return best
