"""Explanation check: does the model look at the right part of the tooth?

For every finding the model detects correctly we compute a heat-map for that detection and
ask a dental question of it:

    caries / deep caries  -> evidence should lie in the CROWN half of the tooth
    periapical lesion     -> evidence should lie in the ROOT half (towards the apex)

On an OPG the upper teeth point down (crown at the bottom of the box, apex at the top)
and the lower teeth point up, so "crown half" depends on the jaw (taken from the FDI
number). Impacted teeth have no single expected region and are reported without a verdict.

Heat-map method: HiResCAM (Draelos & Carin, 2020), a variant of Grad-CAM that multiplies
gradients and feature maps location by location instead of averaging the gradients first.
It shows where the evidence for one detection actually lies, and it does not go blank on
feature maps that take negative values (as the FPN outputs of Faster R-CNN do), which plain
Grad-CAM can. Grad-CAM is available as method="gradcam" for comparison.

Note: in a two-stage detector the classifier only sees features inside the box, so the
heat-map stays inside the box. The test is therefore about WHERE inside the tooth the
model looks, not whether it looks at the right tooth.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torchvision.ops import box_iou

from .agreement import wilson
from .fdi import EXPECTED_REGION, is_upper


class DetectionGradCAM:
    """HiResCAM (default) or Grad-CAM for one detection of a torchvision Faster R-CNN."""

    def __init__(self, model, method="hirescam"):
        assert method in ("hirescam", "gradcam")
        self.model, self.method = model, method
        self._feats, self._padded, self._resized = None, None, None
        self._hooks = [model.backbone.register_forward_hook(self._save_features),
                       model.transform.register_forward_hook(self._save_sizes)]

    def _save_features(self, module, inputs, output):
        self._feats = output
        for v in output.values():
            if v.requires_grad:
                v.retain_grad()

    def _save_sizes(self, module, inputs, output):
        image_list = output[0]
        self._padded = tuple(image_list.tensors.shape[-2:])
        self._resized = tuple(image_list.image_sizes[0])

    def remove(self):
        for h in self._hooks:
            h.remove()

    def __call__(self, image, box=None, label=None, min_iou=0.5, min_score=0.0):
        """Heat-map (H, W) in [0, 1] for the detection that best matches `box` and `label`.

        Only detections with score >= min_score and IoU >= min_iou count. If `box` is None
        the highest-scoring detection is explained. Returns None when nothing matches.
        """
        model = self.model
        model.eval()
        device = next(model.parameters()).device
        with torch.enable_grad():
            out = model([image.to(device)])[0]
            scores = out["scores"]
            if len(scores) == 0:
                return None
            k = 0
            if box is not None:
                target = torch.as_tensor(np.asarray(box, dtype=np.float32), device=device).reshape(1, 4)
                iou = box_iou(out["boxes"].detach(), target)[:, 0]
                ok = scores.detach() >= min_score
                if label is not None:
                    ok = ok & (out["labels"] == int(label))
                iou = torch.where(ok, iou, torch.zeros_like(iou))
                k = int(iou.argmax())
                if float(iou[k]) < min_iou:
                    return None
            score = scores[k]
            model.zero_grad(set_to_none=True)
            score.backward()

        maps = []
        for feat in self._feats.values():
            if feat.grad is None:
                continue
            g, a = feat.grad[0], feat[0].detach()
            if self.method == "gradcam":
                maps.append((g.mean(dim=(1, 2))[:, None, None] * a).sum(0))
            else:
                maps.append((g * a).sum(0))
        if not maps:
            return None
        positive = [torch.relu(m) for m in maps]
        fallback = sum(float(p.sum()) for p in positive) <= 0
        if fallback:  # no positive evidence at all: show where the evidence is, whatever its sign
            positive = [m.abs() for m in maps]
        cam = sum(F.interpolate(p[None, None], size=self._padded, mode="bilinear", align_corners=False)[0, 0]
                  for p in positive)
        h, w = self._resized
        cam = F.interpolate(cam[:h, :w][None, None], size=tuple(image.shape[-2:]), mode="bilinear",
                            align_corners=False)[0, 0]
        cam = (cam / (cam.max() + 1e-8)).detach().cpu().numpy()
        return {"cam": cam, "box": out["boxes"][k].detach().cpu().numpy(), "score": float(score.detach()),
                "label": int(out["labels"][k]), "fallback": fallback}


def crown_root_split(cam, box, fdi):
    """Share of the heat-map inside the box that falls in the crown half of the tooth.

    Also returns peak_depth: 0 = at the occlusal / incisal end, 1 = at the apex end.
    """
    H, W = cam.shape
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
    sub = cam[y0:y1, x0:x1]
    if sub.size == 0 or sub.sum() <= 0:
        return None
    rows = sub.sum(axis=1)
    depth = (np.arange(len(rows)) + 0.5) / len(rows)  # 0 = top of box, 1 = bottom
    if is_upper(fdi):
        depth = 1 - depth  # upper teeth: the crown is at the bottom of the box
    peak_row = np.unravel_index(np.argmax(sub), sub.shape)[0]
    return {"crown_share": float(rows[depth < 0.5].sum() / rows.sum()),
            "peak_depth": float(depth[peak_row])}


def explanation_check(model, ds, find_preds=None, score_thresh=0.3, iou_thresh=0.5, keep_maps=3,
                      method="hirescam", log=print):
    """Run the anatomical plausibility test on all correctly detected findings of a dataset.

    A finding counts as correctly detected when the model finds a box of the same diagnosis
    with score >= score_thresh and IoU >= iou_thresh. find_preds (optional) only speeds things
    up by skipping X-rays without any detection above the threshold.

    Returns (per_finding DataFrame, summary DataFrame, examples) where examples holds a few
    heat-maps per diagnosis for plotting.
    """
    cammer = DetectionGradCAM(model, method=method)
    rows, examples = [], {}
    total = missed = empty = fallback = 0
    try:
        for i, record in enumerate(ds.records):
            gts = [a for a in record["anns"] if a["diagnosis"] in ds.diagnoses and a["fdi"]]
            total += len(gts)
            if not gts:
                continue
            if find_preds is not None and not (find_preds[i]["scores"] >= score_thresh).any():
                missed += len(gts)
                continue
            image = ds.image(i).unsqueeze(0).expand(3, -1, -1).contiguous()
            for a in gts:
                label = ds.diagnoses.index(a["diagnosis"]) + 1
                res = cammer(image, box=a["box"], label=label, min_iou=iou_thresh, min_score=score_thresh)
                if res is None:
                    missed += 1  # not detected at this threshold: nothing to explain
                    continue
                split = crown_root_split(res["cam"], a["box"], a["fdi"])
                if split is None:
                    empty += 1
                    continue
                fallback += res["fallback"]
                expected = EXPECTED_REGION.get(a["diagnosis"])
                plausible = None
                if expected == "crown":
                    plausible = split["crown_share"] > 0.5
                elif expected == "root":
                    plausible = split["crown_share"] < 0.5
                rows.append({"image": record["file_name"], "fdi": a["fdi"], "diagnosis": a["diagnosis"],
                             "score": res["score"], "expected_region": expected or "-", **split,
                             "plausible": plausible})
                ex = examples.setdefault(a["diagnosis"], [])
                if len(ex) < keep_maps:
                    ex.append({"index": i, "box": a["box"], "fdi": a["fdi"], "cam": res["cam"],
                               "crown_share": split["crown_share"]})
    finally:
        cammer.remove()
    log(f"Explained {len(rows)} of {total} true findings ({missed} not detected at score >= {score_thresh:.2f}, "
        f"{empty} with an empty heat-map, {fallback} explained by gradient magnitude).")
    df = pd.DataFrame(rows)
    if df.empty:
        return df, pd.DataFrame(), examples
    summary = (df.groupby("diagnosis")
                 .agg(findings=("fdi", "size"), median_crown_share=("crown_share", "median"),
                      median_peak_depth=("peak_depth", "median"),
                      plausible_rate=("plausible", lambda s: s.dropna().astype(float).mean()
                                      if s.notna().any() else np.nan))
                 .reset_index())
    summary["expected_region"] = summary["diagnosis"].map(lambda d: EXPECTED_REGION.get(d, "-"))
    ci = []
    for dx in summary["diagnosis"]:
        verdicts = df.loc[df["diagnosis"] == dx, "plausible"].dropna().astype(bool)
        lo, hi = wilson(int(verdicts.sum()), len(verdicts))
        ci.append(f"{lo:.2f}-{hi:.2f}" if len(verdicts) else "-")
    summary["plausible_95CI"] = ci
    return df, summary, examples
