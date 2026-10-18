"""Figures: X-rays with boxes, the dental chart, training curves, heat-maps and the study."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from .fdi import CODES, COLORS, LABEL_TO_TOOTH, chart_rows, is_upper


def show_xray(ax, image, boxes=(), texts=(), colors=(), title=None):
    """Draw an X-ray (H, W) with labelled boxes."""
    ax.imshow(image, cmap="gray")
    for b, t, c in zip(boxes, texts, colors):
        ax.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec=c, lw=1.4))
        ax.text(b[0], b[1] - 4, t, color="white", fontsize=7, va="bottom",
                bbox=dict(facecolor=c, alpha=0.85, pad=1, lw=0))
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=10)


def teeth_boxes(pred, score_thresh=0.5):
    """Boxes and FDI texts of the tooth model's prediction for plotting."""
    keep = pred["scores"] >= score_thresh
    return pred["boxes"][keep], [str(LABEL_TO_TOOTH[int(l)]) for l in pred["labels"][keep]]


def plot_dental_chart(findings, present=None, title=None, ax=None):
    """Dental chart in the orientation of the X-ray (patient's right on the left).

    findings: set of (FDI, diagnosis); present: set of FDI numbers (None = not shown).
    """
    upper, lower = chart_rows()
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 2.4))
    by_tooth = {}
    for fdi, dx in findings:
        by_tooth.setdefault(fdi, []).append(dx)
    for row, teeth in enumerate((upper, lower)):
        y = 1 - row
        for col, fdi in enumerate(teeth):
            x = col + (0.3 if col >= 8 else 0)
            missing = present is not None and fdi not in present
            dxs = by_tooth.get(fdi, [])
            face = "#d9d9d9" if missing else (COLORS.get(dxs[0], "#ffffff") if dxs else "#ffffff")
            ax.add_patch(Rectangle((x, y), 0.92, 0.85, fc=face, ec="#555555", lw=0.8))
            ax.text(x + 0.46, y + 0.62, str(fdi), ha="center", va="center", fontsize=7, color="#333333")
            label = "missing" if missing else "/".join(CODES.get(d, d) for d in dxs)
            ax.text(x + 0.46, y + 0.27, label, ha="center", va="center", fontsize=6.5 if missing else 8,
                    color="#555555" if missing else "black", fontweight="bold" if dxs else "normal")
    ax.set_xlim(-0.2, 16.5)
    ax.set_ylim(-0.1, 2.0)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.text(-0.15, 1.42, "R", fontsize=9, ha="right", va="center")
    ax.text(16.45, 1.42, "L", fontsize=9, ha="left", va="center")
    handles = [Rectangle((0, 0), 1, 1, fc=c) for c in COLORS.values()]
    ax.legend(handles, [f"{CODES[d]} = {d.lower()}" for d in COLORS], loc="upper center",
              bbox_to_anchor=(0.5, 0.02), ncol=4, fontsize=7, frameon=False)
    if title:
        ax.set_title(title, fontsize=10)
    return ax


def plot_history(histories):
    """Training loss and validation AP50 per epoch. histories: {name: history list}."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    for name, h in histories.items():
        ep = [r["epoch"] for r in h]
        axes[0].plot(ep, [r["loss"] for r in h], marker="o", ms=3, label=name)
        axes[1].plot(ep, [r["AP50"] for r in h], marker="o", ms=3, label=name)
    axes[0].set(title="Training loss", xlabel="epoch")
    axes[1].set(title="Validation AP50", xlabel="epoch", ylim=(0, 1))
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    return fig


def plot_cam_examples(examples, ds, max_per_row=3):
    """Heat-maps of correctly found findings, cropped around the tooth.

    The dashed line splits the tooth box into crown half and root half.
    """
    names = [d for d in ds.diagnoses if examples.get(d)]
    if not names:
        return None
    fig, axes = plt.subplots(len(names), max_per_row, figsize=(3.2 * max_per_row, 3.4 * len(names)), squeeze=False)
    for r, dx in enumerate(names):
        for c in range(max_per_row):
            ax = axes[r][c]
            ax.set_axis_off()
            if c >= len(examples[dx]):
                continue
            ex = examples[dx][c]
            image = ds.image(ex["index"]).numpy()
            x0, y0, x1, y1 = ex["box"]
            mx, my = 0.6 * (x1 - x0), 0.35 * (y1 - y0)
            cx0, cy0 = int(max(x0 - mx, 0)), int(max(y0 - my, 0))
            cx1, cy1 = int(min(x1 + mx, image.shape[1])), int(min(y1 + my, image.shape[0]))
            ax.imshow(image[cy0:cy1, cx0:cx1], cmap="gray")
            ax.imshow(ex["cam"][cy0:cy1, cx0:cx1], cmap="jet", alpha=0.4, vmin=0, vmax=1)
            ax.add_patch(Rectangle((x0 - cx0, y0 - cy0), x1 - x0, y1 - y0, fill=False, ec="white", lw=1.2))
            ax.plot([x0 - cx0, x1 - cx0], [(y0 + y1) / 2 - cy0] * 2, ls="--", c="white", lw=0.8)
            ax.set_title(f"{ex['fdi']} {dx}\ncrown share {ex['crown_share']:.2f}"
                         f" ({'crown up' if not is_upper(ex['fdi']) else 'crown down'})", fontsize=8)
    fig.tight_layout()
    return fig


def plot_agreement(scores):
    """Sensitivity and PPV per diagnosis for dentist, AI and dentist + AI."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), sharey=True)
    dxs = list(dict.fromkeys(scores["diagnosis"]))
    raters = list(dict.fromkeys(scores["rater"]))
    width = 0.8 / len(raters)
    for ax, metric in zip(axes, ("sensitivity", "PPV")):
        for k, rater in enumerate(raters):
            vals = [float(scores[(scores.diagnosis == d) & (scores.rater == rater)][metric].iloc[0]) for d in dxs]
            ax.bar(np.arange(len(dxs)) + k * width, vals, width, label=rater)
        ax.set_xticks(np.arange(len(dxs)) + width * (len(raters) - 1) / 2)
        ax.set_xticklabels(dxs, rotation=15, fontsize=8)
        ax.set(title=metric, ylim=(0, 1))
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    return fig
