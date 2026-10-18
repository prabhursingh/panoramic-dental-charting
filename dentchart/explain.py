"""Explanation check: does the model look at the right part of the tooth?

For every finding the model detects correctly we compute a Grad-CAM heat-map for that
detection (gradients of its score with respect to the FPN feature maps). We then ask a
dental question of the heat-map:

    caries / deep caries  -> evidence should lie in the CROWN half of the tooth
    periapical lesion     -> evidence should lie in the ROOT half (towards the apex)

On an OPG the upper teeth point down (crown at the bottom of the box, apex at the top)
and the lower teeth point up, so "crown half" depends on the jaw (taken from the FDI
number). Impacted teeth have no single expected region and are reported without a verdict.

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
    """Grad-CAM for one detection of a torchvision Faster R-CNN."""

    def __init__(self, model):
        self.model = model
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

    def __call__(self, image, box=None, label=None, min_iou=0.5):
        """Heat-map (H, W) in [0, 1] for the detection that matches `box` and `label`.

        If `box` is None the highest-scoring detection is explained.
        Returns None when no matching detection exists.
        """
        model = self.model
        model.eval()
        device = next(model.parameters()).device
        with torch.enable_grad():
            out = model([image.to(device)])[0]
            if len(out["scores"]) == 0:
                return None
            k = 0
            if box is not None:
                iou = box_iou(out["boxes"].detach(), torch.as_tensor(box, dtype=torch.float32, device=device)[None])[:, 0]
                if label is not None:
                    iou = iou * (out["labels"] == int(label))
                k = int(iou.argmax())
                if float(iou[k]) < min_iou:
                    return None
            score = out["scores"][k]
            model.zero_grad(set_to_none=True)
            score.backward()

        cam = None
        for feat in self._feats.values():
            if feat.grad is None:
                continue
            weights = feat.grad[0].mean(dim=(1, 2))
            level = torch.relu((weights[:, None, None] * feat[0].detach()).sum(0))
            level = F.interpolate(level[None, None], size=self._padded, mode="bilinear", align_corners=False)[0, 0]
            cam = level if cam is None else cam + level
        if cam is None:
            return None
        h, w = self._resized
        cam = F.interpolate(cam[:h, :w][None, None], size=tuple(image.shape[-2:]), mode="bilinear",
                            align_corners=False)[0, 0]
        cam = (cam / (cam.max() + 1e-8)).detach().cpu().numpy()
        return {"cam": cam, "box": out["boxes"][k].detach().cpu().numpy(),
                "score": float(score.detach()), "label": int(out["labels"][k])}


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


def explanation_check(model, ds, find_preds, score_thresh=0.3, iou_thresh=0.5, keep_maps=3, log=print):
    """Run the anatomical plausibility test on all correctly detected findings of a dataset.

    Returns (per_finding DataFrame, summary DataFrame, examples) where examples holds a few
    heat-maps per diagnosis for plotting.
    """
    cammer = DetectionGradCAM(model)
    rows, examples = [], {}
    try:
        for i, pred in enumerate(find_preds):
            record = ds.records[i]
            gts = [a for a in record["anns"] if a["diagnosis"] in ds.diagnoses and a["fdi"]]
            keep = pred["scores"] >= score_thresh
            if not gts or not keep.any():
                continue
            boxes, labels = pred["boxes"][keep], pred["labels"][keep]
            image = None
            for a in gts:
                label = ds.diagnoses.index(a["diagnosis"]) + 1
                iou = box_iou(torch.as_tensor([a["box"]], dtype=torch.float32), torch.as_tensor(boxes))[0]
                iou = iou * torch.as_tensor(labels == label)
                if float(iou.max()) < iou_thresh:
                    continue  # missed finding: nothing to explain
                if image is None:
                    image = ds.image(i).unsqueeze(0).expand(3, -1, -1).contiguous()
                res = cammer(image, box=boxes[int(iou.argmax())], label=label)
                if res is None:
                    continue
                split = crown_root_split(res["cam"], a["box"], a["fdi"])
                if split is None:
                    continue
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
    df = pd.DataFrame(rows)
    if df.empty:
        log("No correctly detected findings to explain yet.")
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
