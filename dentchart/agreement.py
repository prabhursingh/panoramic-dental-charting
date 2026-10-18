"""Dentist-versus-AI study on the 50 official DENTEX validation X-rays.

1. export_study()   copies the X-rays and creates an empty chart (my_chart.csv)
2. charting_tool()  lets the dentist chart each X-ray inside the notebook
                    (or fill my_chart.csv in Excel / Google Sheets)
3. compare()        scores dentist, AI and dentist + AI against the DENTEX ground truth

Chart format, one line per X-ray: FDI number + code, separated by ';'
    C = caries, DC = deep caries, PA = periapical lesion, I = impacted
    e.g.  "36 DC; 48 I; 26 C"     (leave empty if no abnormal tooth)
"""

import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .data import read_xray
from .fdi import CODE_TO_DIAGNOSIS, DIAGNOSES, TEETH

INSTRUCTIONS = """How to chart the X-rays
=======================

Look at each panoramic X-ray as you would in the clinic and list every ABNORMAL tooth.
Write the FDI number and a code, separated by ';':

    C  = caries
    DC = deep caries (close to or into the pulp)
    PA = periapical lesion
    I  = impacted tooth

Example:  36 DC; 48 I; 26 C
Leave the line empty if you see no abnormal tooth.

Please chart BEFORE looking at the AI results or the DENTEX labels, so the
comparison is fair (blinded). Take your time, but do not go back to change
answers after seeing the AI.
"""

_WORDS = {"C": "Caries", "CARIES": "Caries", "DC": "Deep Caries", "DEEPCARIES": "Deep Caries",
          "PA": "Periapical Lesion", "PERIAPICAL": "Periapical Lesion",
          "PERIAPICALLESION": "Periapical Lesion", "I": "Impacted", "IMP": "Impacted",
          "IMPACTED": "Impacted"}


def parse_findings(text):
    """'36 DC; 48 I' -> ({(36, 'Deep Caries'), (48, 'Impacted')}, errors)."""
    found, errors = set(), []
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return found, errors
    for token in re.split(r"[;,\n]+", str(text)):
        token = token.strip()
        if not token:
            continue
        m = re.match(r"^(\d{2})\s*[-:=/ ]?\s*([A-Za-z][A-Za-z ]*)$", token)
        code = m.group(2).replace(" ", "").upper() if m else ""
        dx = _WORDS.get(code) or CODE_TO_DIAGNOSIS.get(code)
        if not m or int(m.group(1)) not in TEETH or dx is None:
            errors.append(token)
            continue
        found.add((int(m.group(1)), dx))
    return found, errors


def export_study(records, image_dir, out_dir):
    """Copy the study X-rays and create my_chart.csv (an existing chart is never overwritten)."""
    out = Path(out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    names = [r["file_name"] for r in records]
    for n in names:
        dst = out / "images" / n
        if not dst.exists():
            shutil.copy(Path(image_dir) / n, dst)
    chart = out / "my_chart.csv"
    if not chart.exists():
        pd.DataFrame({"image": names, "findings": [""] * len(names),
                      "reviewed": ["no"] * len(names)}).to_csv(chart, index=False)
    (out / "HOW_TO_CHART.txt").write_text(INSTRUCTIONS)
    return chart


def read_chart(csv_path):
    """Read the dentist's chart. Returns ({image: set of (FDI, diagnosis)}, problems)."""
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    charts, problems = {}, []
    for _, row in df.iterrows():
        reviewed = str(row.get("reviewed", "yes")).strip().lower() in ("yes", "y", "1", "true", "done")
        if not reviewed:
            continue
        found, errors = parse_findings(row.get("findings", ""))
        charts[row["image"]] = found
        problems += [f"{row['image']}: could not read '{e}'" for e in errors]
    return charts, problems


def wilson(k, n, z=1.96):
    """95% Wilson confidence interval for a proportion k / n (robust for small n)."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def _units(sets_by_image, images, dx):
    """Binary vector over (X-ray, tooth) units for one diagnosis."""
    return np.array([int((fdi, dx) in sets_by_image.get(img, set())) for img in images for fdi in TEETH])


def _kappa(a, b):
    from sklearn.metrics import cohen_kappa_score
    if a.sum() == 0 and b.sum() == 0:
        return np.nan
    return float(cohen_kappa_score(a, b))


def compare(gt, dentist, ai, diagnoses=DIAGNOSES):
    """Score the dentist, the AI and the two together against the DENTEX ground truth.

    gt, dentist, ai: {image: set of (FDI, diagnosis)}. Only X-rays the dentist reviewed count.
    Returns (scores DataFrame, dentist-vs-AI agreement DataFrame).
    """
    images = sorted(dentist)
    raters = {"Dentist": dentist, "AI": ai,
              "Dentist + AI": {i: dentist.get(i, set()) | ai.get(i, set()) for i in images}}
    rows, pair = [], []
    for dx in list(diagnoses) + ["All"]:
        dxs = list(diagnoses) if dx == "All" else [dx]
        g = np.concatenate([_units(gt, images, d) for d in dxs])
        for name, sets in raters.items():
            r = np.concatenate([_units(sets, images, d) for d in dxs])
            tp, fp, fn = int(((r == 1) & (g == 1)).sum()), int(((r == 1) & (g == 0)).sum()), int(((r == 0) & (g == 1)).sum())
            sens, ppv = tp / max(tp + fn, 1), tp / max(tp + fp, 1)
            lo, hi = wilson(tp, tp + fn)
            rows.append({"diagnosis": dx, "rater": name, "TP": tp, "FP": fp, "FN": fn,
                         "sensitivity": sens, "sens_95CI": f"{lo:.2f}-{hi:.2f}" if tp + fn else "-",
                         "PPV": ppv, "F1": 2 * sens * ppv / max(sens + ppv, 1e-9),
                         "kappa_vs_truth": _kappa(r, g)})
        d = np.concatenate([_units(dentist, images, x) for x in dxs])
        a = np.concatenate([_units(ai, images, x) for x in dxs])
        pair.append({"diagnosis": dx, "kappa_dentist_vs_AI": _kappa(d, a),
                     "both_found": int(((d == 1) & (a == 1)).sum()),
                     "only_dentist": int(((d == 1) & (a == 0)).sum()),
                     "only_AI": int(((d == 0) & (a == 1)).sum())})
    return pd.DataFrame(rows), pd.DataFrame(pair)


def charting_tool(image_dir, csv_path, display_width=2000):
    """Small in-notebook tool: one X-ray at a time, type the findings, press Next.

    Answers are saved to csv_path after every step, so you can stop and continue later.
    """
    import io

    import ipywidgets as widgets
    from IPython.display import display
    from PIL import Image

    try:  # Colab needs this for some widget versions; harmless elsewhere
        from google.colab import output
        output.enable_custom_widget_manager()
    except Exception:
        pass

    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    state = {"i": next((k for k, v in enumerate(df["reviewed"]) if v.lower() != "yes"), 0)}
    picture = widgets.Image(format="jpeg", layout=widgets.Layout(width="100%"))
    entry = widgets.Text(placeholder="e.g. 36 DC; 48 I   (C, DC, PA, I) - leave empty if none",
                         layout=widgets.Layout(width="60%"))
    status = widgets.HTML()
    back, nxt = widgets.Button(description="< Back"), widgets.Button(description="Save & next >",
                                                                        button_style="primary")

    def show():
        i = state["i"]
        img = Image.fromarray((read_xray(Path(image_dir) / df.at[i, "image"]) * 255).astype(np.uint8))
        if img.width > display_width:
            img = img.resize((display_width, round(img.height * display_width / img.width)))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=92)
        picture.value = buf.getvalue()
        entry.value = df.at[i, "findings"]
        done = int((df["reviewed"].str.lower() == "yes").sum())
        status.value = f"<b>X-ray {i + 1} of {len(df)}</b> ({df.at[i, 'image']}) &nbsp; charted so far: {done}/{len(df)}"

    def save():
        i = state["i"]
        _, errors = parse_findings(entry.value)
        if errors:
            status.value = f"<span style='color:#c00'>Could not read: {', '.join(errors)}. Use e.g. 36 DC; 48 I</span>"
            return False
        df.at[i, "findings"], df.at[i, "reviewed"] = entry.value.strip(), "yes"
        df.to_csv(csv_path, index=False)
        return True

    def on_next(_):
        if save():
            state["i"] = min(state["i"] + 1, len(df) - 1)
            show()

    def on_back(_):
        state["i"] = max(state["i"] - 1, 0)
        show()

    nxt.on_click(on_next)
    back.on_click(on_back)
    show()
    display(widgets.VBox([status, picture, widgets.HBox([back, entry, nxt])]))
