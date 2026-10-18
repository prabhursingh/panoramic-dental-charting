"""Check the scoring code with perfect ("oracle") predictions: every score must be perfect.

    python -m pytest tests -q        or        python tests/test_oracle.py
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dentchart import data, evaluate, explain, fdi  # noqa: E402


def _fake_dentex():
    tmp = Path(tempfile.mkdtemp())
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_synthetic_dentex.py"), "--out", str(tmp / "raw"),
                    "--n", "10", "--n-val", "3"], check=True, capture_output=True)
    data.prepare_dentex(tmp / "raw" / "training_data.zip", tmp / "d", log=lambda *a: None)
    return tmp / "d"


def _oracle(ds):
    preds = []
    for i in range(len(ds)):
        boxes, labels, _ = ds.annotations(i)
        preds.append({"boxes": boxes.numpy(), "scores": np.ones(len(labels), dtype=np.float32),
                      "labels": labels.numpy()})
    return preds


def test_perfect_predictions_score_perfectly():
    root = _fake_dentex()
    recs, _ = data.load_records(root / "enumeration")
    teeth = data.OPGDataset(recs, root / "enumeration" / "images", "teeth")
    preds = _oracle(teeth)
    m = evaluate.coco_metrics(preds, teeth)
    assert m["AP50"] > 0.99 and m["AP"] > 0.99
    c = evaluate.teeth_chart_metrics(preds, teeth)
    assert c["chart_accuracy"] == 1 and c["numbering_accuracy"] == 1 and c["detection_rate"] == 1
    assert c["missing_sensitivity"] == 1

    dis, dxs = data.load_records(root / "disease")
    find = data.OPGDataset(dis, root / "disease" / "images", "findings", dxs)
    on_teeth = data.OPGDataset(dis, root / "disease" / "images", "teeth")
    fp, tp = _oracle(find), _oracle(on_teeth)
    gts = [evaluate.gt_findings(r, dxs) for r in dis]
    sets = [evaluate.chart_findings(f, t, dxs, find_thresh=0.5) for f, t in zip(fp, tp)]
    assert sets == gts
    assert evaluate.chart_scores(sets, gts, dxs)["All"]["F1"] == 1


def test_crown_root_orientation():
    cam = np.zeros((100, 100), dtype=np.float32)
    cam[80:95, 40:60] = 1.0  # energy at the bottom of the box
    box = [30, 10, 70, 100]
    lower = explain.crown_root_split(cam, box, 36)  # lower tooth: crown on top -> bottom is root
    upper = explain.crown_root_split(cam, box, 16)  # upper tooth: crown at the bottom
    assert lower["crown_share"] < 0.05 and lower["peak_depth"] > 0.7
    assert upper["crown_share"] > 0.95 and upper["peak_depth"] < 0.3


if __name__ == "__main__":
    test_perfect_predictions_score_perfectly()
    test_crown_root_orientation()
    print("oracle tests passed")
