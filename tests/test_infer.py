"""Charting new X-rays (dentchart/infer.py) and the dashboard page (app/dashboard.py).

A fake detector stands in for the trained models: it finds the bright "teeth" on a fake OPG
and, like a real detector, is only confident when they look as big as in its training X-rays.
That tests the X-ray preparation, the scale fit and the mapping of boxes back to the X-ray.

    python -m pytest tests -q        or        python tests/test_infer.py
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dentchart import fdi  # noqa: E402
from dentchart import infer  # noqa: E402

STRIP = 60  # black label band on the left, like the VistaPano export


def fake_opg(seed=0):
    """Fake OPG 2000 x 930 px (DENTEX-like shape) with 28 teeth, plus a black label band with an "R"."""
    rng = np.random.default_rng(seed)
    h, w = 930, 2000
    img = np.clip(rng.normal(0.35, 0.03, (h, w)), 0, 1).astype(np.float32)
    boxes = {}
    upper, lower = fdi.chart_rows()
    for teeth, (y0, y1) in ((upper, (100, 475)), (lower, (495, 870))):
        for i, t in enumerate(teeth):
            if t % 10 == 8:
                continue  # wisdom teeth missing
            x0 = 120 + 110 * i
            img[y0:y1, x0:x0 + 90] = 0.85
            boxes[t] = [x0 + STRIP, y0, x0 + 90 + STRIP, y1]
    full = np.zeros((h, w + STRIP), np.float32)
    full[:, STRIP:] = img
    full[300:330, 10:50] = 1.0   # "R" marker
    full[600:615, 5:55] = 0.9    # machine name
    return full, boxes


class _Transform:
    def __init__(self, max_size):
        self.min_size, self.max_size = (800,), max_size


class FakeDetector:
    """Boxes the bright bars; confident only when they appear at `size` px after its own resize."""

    def __init__(self, max_size, size, label=None, tolerance=0.25):
        self.transform, self.size, self.label, self.tolerance = _Transform(max_size), size, label, tolerance

    def __call__(self, images):
        x = images[0][0].numpy()
        scale = infer.own_scale(self, *x.shape)
        boxes, scores, labels = [], [], []
        for i, (ys, xs) in enumerate(ndimage.find_objects(ndimage.label(x > 0.6)[0])):
            bw, bh = (xs.stop - xs.start) * scale, (ys.stop - ys.start) * scale
            d = np.log(bw / self.size[0]) ** 2 + np.log(bh / self.size[1]) ** 2
            boxes.append([xs.start, ys.start, xs.stop, ys.stop])
            scores.append(float(np.exp(-d / (2 * self.tolerance ** 2))))
            labels.append(self.label or i % 32 + 1)
        return [{"boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
                 "scores": torch.tensor(scores, dtype=torch.float32),
                 "labels": torch.tensor(labels, dtype=torch.int64)}]


def _models():
    # on a DENTEX-shaped X-ray the tooth model sees teeth at 60 x 250 px and the findings model
    # (trained on 1600 px wide images) at 72 x 300 px; the findings fake calls every tooth "Caries"
    return FakeDetector(1333, (60, 250)), FakeDetector(1600, (72, 300), label=2)


def _check_boxes(result, truth, sx=1.0, sy=1.0):
    pred = torch.as_tensor(np.asarray(result["teeth"]["boxes"]))
    gt = torch.tensor([[b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy] for b in truth.values()])
    from torchvision.ops import box_iou
    best = box_iou(gt, pred).max(dim=1).values
    assert len(pred) == len(gt) and float(best.min()) > 0.85, best.min()


def test_to_grey_matches_training_scaling():
    arr = np.linspace(0, 1, 60 * 80, dtype=np.float32).reshape(60, 80)
    u8 = (arr * 255).round().astype(np.uint8)
    for x in (u8, np.stack([u8] * 3, -1), torch.from_numpy(np.stack([arr] * 3)), Image.fromarray(u8).convert("RGB")):
        g = infer.to_grey(x)
        assert g.shape == (60, 80) and abs(float(g.max()) - 1) < 0.01 and float(g.min()) == 0
    g16 = infer.to_grey(Image.fromarray((arr * 4095).astype(np.uint16)))   # 12-bit values in a 16-bit file
    assert abs(float(g16.max()) - 1) < 1e-6


def test_label_band_and_white_frame_are_cropped():
    full, _ = fake_opg()
    x0, y0, x1, y1 = infer.find_xray_area(full)
    assert (x0, y0, x1, y1) == (STRIP, 0, full.shape[1], full.shape[0])
    framed = np.pad(full, 25, constant_values=1.0)                         # white frame around it all
    assert infer.find_xray_area(framed) == (25 + STRIP, 25, 25 + full.shape[1], 25 + full.shape[0])


def test_dentex_like_xray_is_used_as_it_is():
    full, truth = fake_opg()
    teeth, find = _models()
    r = infer.chart_opg(full, teeth, find)
    assert r["setting"] == {"stretch": 1.0, "zoom": 1.0, "contrast": False, "scale": None}
    assert len(r["tried"]) == 1 and len(r["present"]) == 28 and not r["warnings"]
    _check_boxes(r, truth)
    assert len(r["findings"]) == 28 and not r["unassigned"]                # every finding on its tooth


def test_squeezed_xray_is_stretched_back():
    full, truth = fake_opg()
    w = 900
    squeezed = np.asarray(Image.fromarray(full).resize((w, full.shape[0]), Image.BILINEAR))
    teeth, find = _models()
    r = infer.chart_opg(squeezed, teeth, find)
    assert r["setting"]["stretch"] > 2 and r["setting"]["zoom"] == 1.0, r["tried"]
    assert len(r["present"]) == 28 and len(r["findings"]) == 28
    _check_boxes(r, truth, sx=w / full.shape[1])                            # boxes on the squeezed image
    assert r["warnings"]                                                    # and the user is told


def test_squeezed_xray_is_stretched_even_if_teeth_are_found():
    # a tolerant tooth model still finds every tooth on the squeezed X-ray, but less confidently:
    # the unusual shape must still trigger the search, and the stretched setting must win
    full, truth = fake_opg()
    squeezed = np.asarray(Image.fromarray(full).resize((900, full.shape[0]), Image.BILINEAR))
    teeth, find = _models()
    teeth.tolerance = 0.9
    r = infer.chart_opg(squeezed, teeth, find)
    as_is = r["tried"][0]
    assert as_is["teeth"] == 28 and len(r["tried"]) > 1
    assert r["setting"]["stretch"] > 2, r["tried"]


def test_cropped_xray_is_zoomed_out():
    full, truth = fake_opg()
    half = full[:, :1060]                                                   # band + patient's right side
    teeth, find = _models()
    r = infer.chart_opg(half, teeth, find)
    assert r["setting"]["zoom"] < 1 and r["setting"]["stretch"] == 1.0, r["tried"]
    inside = {t: b for t, b in truth.items() if b[2] <= 1060}
    _check_boxes(r, inside)
    assert len(r["findings"]) == len(inside)                                # findings model at its own scale too


def test_no_teeth_gives_a_warning():
    rng = np.random.default_rng(1)
    noise = np.clip(rng.normal(0.4, 0.05, (500, 1000)), 0, 1).astype(np.float32)
    teeth, find = _models()
    r = infer.chart_opg(noise, teeth, find)
    assert len(r["present"]) == 0 and any("do not trust" in w for w in r["warnings"])
    assert len(r["tried"]) == 6                                             # 3 zooms x 2 contrasts


def _fake_checkpoints(work):
    from dentchart.model import build_model
    for name, n in (("teeth", 33), ("findings", 5)):
        (work / "runs" / name).mkdir(parents=True)
        torch.save({"model": build_model(n, pretrained=False).state_dict(), "epoch": 3, "val": {"AP50": 0.5}},
                   work / "runs" / name / "best.pt")


def test_models_load_strictly_or_not_at_all():
    tmp = Path(tempfile.mkdtemp())
    with pytest.raises(FileNotFoundError):                    # no checkpoint: an error, not random weights
        infer.load_models([tmp / "missing"])
    _fake_checkpoints(tmp / "work")
    teeth, find, device, thresh, info = infer.load_models([tmp / "missing", tmp / "work"])
    assert "epoch 3" in info and thresh == 0.5 and device in ("cpu", "cuda")
    teeth_ck = tmp / "work" / "runs" / "teeth" / "best.pt"
    ck = torch.load(teeth_ck, weights_only=False)
    torch.save(torch.load(tmp / "work" / "runs" / "findings" / "best.pt", weights_only=False), teeth_ck)
    with pytest.raises(ValueError):                           # swapped checkpoints
        infer.load_models([tmp / "work"])
    del ck["model"]["backbone.fpn.layer_blocks.0.0.weight"]
    torch.save(ck, teeth_ck)
    with pytest.raises(RuntimeError):                         # incomplete or other architecture
        infer.load_models([tmp / "work"])


def test_dashboard_page():
    testing = pytest.importorskip("streamlit.testing.v1")
    tmp = Path(tempfile.mkdtemp())
    app = str(ROOT / "app" / "dashboard.py")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("DENTCHART_MODEL_DIRS", str(tmp / "missing"))
        at = testing.AppTest.from_file(app, default_timeout=300).run()
        assert not at.exception and at.error and not at.success      # says so instead of running untrained
        _fake_checkpoints(tmp / "work")
        mp.setenv("DENTCHART_MODEL_DIRS", str(tmp / "work"))
        at = testing.AppTest.from_file(app, default_timeout=300).run()
        assert not at.exception and not at.error and "epoch 3" in at.success[0].value


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("infer tests passed")
