"""End-to-end smoke test on a tiny fake DENTEX download (CPU, about 2-4 minutes).

    python -m pytest tests -q        or        python tests/test_pipeline.py
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dentchart import agreement, data, evaluate, explain, fdi, model, train, viz  # noqa: E402


def test_fdi_helpers():
    assert fdi.fdi_from_dentex(3, 7) == 48
    assert fdi.mirror(36) == 46 and fdi.mirror(11) == 21 and fdi.mirror(fdi.mirror(27)) == 27
    assert len(fdi.TEETH) == 32 and fdi.is_upper(17) and not fdi.is_upper(31)
    found, errors = agreement.parse_findings("36 DC; 48 I, 26c; 99 X; 11 periapical lesion")
    assert found == {(36, "Deep Caries"), (48, "Impacted"), (26, "Caries"), (11, "Periapical Lesion")}
    assert errors == ["99 X"]
    assert agreement.parse_findings("none") == (set(), []) and agreement.parse_findings(" - ") == (set(), [])


def test_pipeline():
    tmp = Path(tempfile.mkdtemp())
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_synthetic_dentex.py"), "--out", str(tmp / "raw")],
                   check=True)
    summary = data.prepare_dentex(tmp / "raw" / "training_data.zip", tmp / "dentex",
                                  val_zip=tmp / "raw" / "validation_data.zip",
                                  val_json=tmp / "raw" / "validation_triple.json")
    assert summary["enumeration"]["images"] == 24 and summary["disease"]["images"] == 24
    assert summary["validation"]["images_found"] == 8

    teeth_recs, _ = data.load_records(tmp / "dentex" / "enumeration")
    dis_recs, diagnoses = data.load_records(tmp / "dentex" / "disease")
    assert diagnoses == fdi.DIAGNOSES
    tr, va, te = data.split_records(teeth_recs)
    assert len(tr) + len(va) + len(te) == 24 and not set(r["file_name"] for r in te) & set(r["file_name"] for r in tr)

    # the left-right flip must mirror both the boxes and the tooth numbers
    flip_ds = data.OPGDataset(tr, tmp / "dentex" / "enumeration" / "images", "teeth", train=True, flip_p=1.0, jitter=0)
    img, tgt = flip_ds[0]
    boxes, labels, fdis = flip_ds.annotations(0)
    width = img.shape[-1]
    k = 0
    assert fdi.LABEL_TO_TOOTH[int(tgt["labels"][k])] == fdi.mirror(fdis[k])
    assert torch.allclose(tgt["boxes"][k], torch.tensor([width - boxes[k, 2], boxes[k, 1], width - boxes[k, 0], boxes[k, 3]]))

    img_dir = tmp / "dentex"
    t_train = data.OPGDataset(tr, img_dir / "enumeration" / "images", "teeth", train=True)
    t_val = data.OPGDataset(va, img_dir / "enumeration" / "images", "teeth")
    t_test = data.OPGDataset(te, img_dir / "enumeration" / "images", "teeth")
    teeth_model = model.build_model(t_train.num_classes, pretrained=False, min_size=300, max_size=640)
    hist = train.train_detector(teeth_model, t_train, t_val, tmp / "runs" / "teeth", epochs=1, batch_size=2,
                                lr=0.005, num_workers=0, device="cpu")
    assert len(hist) == 1 and (tmp / "runs" / "teeth" / "best.pt").exists()
    preds = evaluate.predict(teeth_model, t_test, device="cpu", num_workers=0)
    assert set(evaluate.coco_metrics(preds, t_test)) >= {"AP", "AP50", "AP50_per_class"}
    assert 0 <= evaluate.teeth_chart_metrics(preds, t_test)["chart_accuracy"] <= 1

    d_tr, d_va, d_te = data.split_records(dis_recs)
    f_train = data.OPGDataset(d_tr, img_dir / "disease" / "images", "findings", diagnoses, train=True)
    f_val = data.OPGDataset(d_va, img_dir / "disease" / "images", "findings", diagnoses)
    f_test = data.OPGDataset(d_te, img_dir / "disease" / "images", "findings", diagnoses)
    find_model = model.build_model(f_train.num_classes, pretrained=False, min_size=300, max_size=640)
    train.train_detector(find_model, f_train, f_val, tmp / "runs" / "findings", epochs=1, batch_size=2,
                         lr=0.005, num_workers=0, device="cpu")
    # resuming a finished run must not train again
    train.train_detector(find_model, f_train, f_val, tmp / "runs" / "findings", epochs=1, num_workers=0, device="cpu")

    teeth_on_test = data.OPGDataset(d_te, img_dir / "disease" / "images", "teeth")
    fp = evaluate.predict(find_model, f_test, device="cpu", num_workers=0)
    tp = evaluate.predict(teeth_model, teeth_on_test, device="cpu", num_workers=0)
    gts = [evaluate.gt_findings(r, diagnoses) for r in d_te]
    thr, _ = evaluate.best_threshold(fp, tp, gts, diagnoses)
    sets = [evaluate.chart_findings(f, t, diagnoses, find_thresh=thr) for f, t in zip(fp, tp)]
    scores = evaluate.chart_scores(sets, gts, diagnoses)
    assert "All" in scores and scores["All"]["TP"] + scores["All"]["FN"] == sum(len(g) for g in gts)

    # explanation check: lower the threshold so the untrained model has something to explain
    find_model.roi_heads.score_thresh = 0.0
    fp0 = evaluate.predict(find_model, f_test, device="cpu", num_workers=0)
    cam = explain.DetectionGradCAM(find_model)
    res = cam(f_test[0][0])
    cam.remove()
    assert res is not None and res["cam"].shape == tuple(f_test[0][0].shape[-2:]) and res["cam"].max() <= 1.0001
    split = explain.crown_root_split(res["cam"], res["box"], 36)
    assert split is None or 0 <= split["crown_share"] <= 1
    df, summary, examples = explain.explanation_check(find_model, f_test, fp0, score_thresh=0.0, iou_thresh=0.1)
    if examples:
        viz.plot_cam_examples(examples, f_test).savefig(tmp / "cams.png")

    # dentist study on the validation X-rays, with a fake dentist who copies part of the truth
    val_recs, _ = data.load_records(tmp / "dentex" / "validation", diagnoses)
    chart_csv = agreement.export_study(val_recs, tmp / "dentex" / "validation" / "images", tmp / "study")
    import pandas as pd
    df_chart = pd.read_csv(chart_csv, dtype=str, keep_default_na=False)
    for i, r in enumerate(val_recs):
        truth = sorted(evaluate.gt_findings(r, diagnoses))[: 1 + i % 2]
        df_chart.loc[df_chart.image == r["file_name"], "findings"] = "; ".join(
            f"{t} {fdi.CODES[d]}" for t, d in truth)
        df_chart.loc[df_chart.image == r["file_name"], "reviewed"] = "yes"
    df_chart.to_csv(chart_csv, index=False)
    dentist, problems = agreement.read_chart(chart_csv)
    assert not problems and len(dentist) == len(val_recs)
    v_find = data.OPGDataset(val_recs, tmp / "dentex" / "validation" / "images", "findings", diagnoses)
    v_teeth = data.OPGDataset(val_recs, tmp / "dentex" / "validation" / "images", "teeth")
    vf = evaluate.predict(find_model, v_find, device="cpu", num_workers=0)
    vt = evaluate.predict(teeth_model, v_teeth, device="cpu", num_workers=0)
    ai = {r["file_name"]: evaluate.chart_findings(f, t, diagnoses, find_thresh=thr) for r, f, t in zip(val_recs, vf, vt)}
    gt = {r["file_name"]: evaluate.gt_findings(r, diagnoses) for r in val_recs}
    table, pair = agreement.compare(gt, dentist, ai, diagnoses)
    dent_all = table[(table.diagnosis == "All") & (table.rater == "Dentist")].iloc[0]
    assert dent_all["PPV"] == 1.0 and dent_all["FP"] == 0
    viz.plot_agreement(table).savefig(tmp / "agreement.png")
    viz.plot_dental_chart(gt[val_recs[0]["file_name"]], present=set(fdi.TEETH) - {18, 28}).figure.savefig(tmp / "chart.png")
    viz.plot_history({"teeth": hist}).savefig(tmp / "history.png")
    print("smoke test passed:", tmp)


if __name__ == "__main__":
    test_fdi_helpers()
    test_pipeline()
