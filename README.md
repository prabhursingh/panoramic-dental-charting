<div align="center">

# 🦷 Dental Charting on Panoramic X-rays

### Tooth numbering, findings, an explanation check and a dentist-versus-AI study on the DENTEX benchmark

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/prabhursingh/panoramic-dental-charting/blob/main/notebooks/dental_charting_colab.ipynb)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Faster%20R--CNN-EE4C2C?logo=pytorch&logoColor=white)
![Explainability](https://img.shields.io/badge/XAI-Grad--CAM-6a5acd)
![Data](https://img.shields.io/badge/data-DENTEX%20(MICCAI%202023)-0B5394)
![License](https://img.shields.io/badge/code-MIT-green)

</div>

> **Status:** code complete and tested end-to-end on synthetic data. The training run on the real
> DENTEX X-rays and the blinded dentist study are in progress; results will be added below.

## 📌 Why this project

Panoramic X-rays (OPGs) are taken at almost every dental check-up, and reading them takes time:
small caries and periapical lesions are easy to miss. An AI that charts an OPG the way a dentist
does could act as a second reader. But for clinical use, accuracy alone is not enough. A dentist
also needs to know **whether the AI looks at the right anatomy** and **how it compares with a
dentist**.

This project therefore asks three questions on real clinical X-rays:

1. **Charting:** can the AI number every tooth (FDI), list missing teeth, and place each finding
   on the chart, for example *36 – deep caries*?
2. **Explanations:** are its heat-maps anatomically plausible: on the **crown** for caries, on
   the **root apex** for periapical lesions?
3. **Clinical validation:** how does it compare with a dentist who charts the same X-rays blinded,
   and does it help as a second reader?

It is the 2D groundwork for dental AI on 3D intraoral scans, where the same three questions
(charting, explainability, validation against clinical findings) apply.

## ⚙️ Pipeline

```
                ┌─► Tooth model (Faster R-CNN, 32 FDI classes) ──► dental chart: present / missing teeth
 Panoramic ─────┤                                                        │
  X-ray         └─► Findings model (Faster R-CNN, 4 diagnoses) ──► each finding mapped to its tooth: "36 DC"
                                          │
                                          └─► Grad-CAM ──► anatomical check: crown (caries) vs apex (periapical lesion)

 50 blinded study X-rays ──► dentist chart  vs  AI chart  vs  DENTEX ground truth
                              (sensitivity, PPV, Cohen's kappa, AI as a second reader)
```

## 🗂️ Data

[DENTEX](https://github.com/ibrahimethemhamamci/DENTEX) (MICCAI 2023): panoramic X-rays from three
institutions with different equipment, labelled in a hierarchy (quadrant → tooth → diagnosis).

| Set used here | X-rays | Labels | Role |
|---|---:|---|---|
| Quadrant–enumeration | 634 | every tooth boxed with its FDI number | tooth model (75 / 10 / 15 % train / val / test) |
| Quadrant–enumeration–diagnosis | 705 | abnormal teeth: FDI number + caries, deep caries, periapical lesion, impacted | findings model (75 / 10 / 15 %) |
| Official validation set | 50 | as above | **blinded dentist study only**, never used for training |

> 🔒 No X-rays are stored in this repository. DENTEX is licensed **CC BY-NC-SA 4.0** (non-commercial);
> the notebook downloads it from [Hugging Face](https://huggingface.co/datasets/ibrahimhamamci/DENTEX).

## 🔬 Methods

**Models.** Two Faster R-CNN detectors (ResNet-50 FPN v2, COCO-pretrained, fine-tuned with SGD,
warm-up and cosine learning rate, mixed precision; best epoch chosen on validation AP50).

**Dental detail in the augmentation.** X-rays are randomly flipped left–right during training.
A flip moves teeth to the other side of the mouth, so FDI numbers are mirrored too (36 → 46).
Flipping without relabelling would teach the model the wrong side.

**From boxes to a chart.** Each finding is assigned the FDI number of the tooth box it overlaps
most. A charted finding counts as correct only if **both** the tooth number and the diagnosis are
right. The score threshold for findings is chosen on the validation X-rays, never on the test set.

**Explanation check.** For every correctly detected finding, a Grad-CAM heat-map is computed for
that detection. The tooth box is split into crown half and root half, oriented by jaw (upper teeth
point down on an OPG, lower teeth point up). A heat-map counts as *anatomically plausible* if most
of it lies in the crown for caries and deep caries, and in the root half for periapical lesions.

**Dentist study.** A prosthodontist with 8+ years of clinical experience charts the 50 official
validation X-rays in the notebook's charting tool, blinded to labels and AI output. Every
(X-ray × tooth × diagnosis) is one unit. Dentist, AI and *dentist + AI* (AI as a second reader)
are scored against the DENTEX ground truth with sensitivity, PPV and Cohen's kappa.

## 📊 Results

*To be added after the training run on the real data.*

| Question | Metric | Result |
|---|---|---|
| Tooth numbering | AP50 · correct FDI number · missing teeth recognised | – |
| Findings on the chart | precision · recall · F1 (tooth **and** diagnosis correct) | – |
| Explanation check | heat-map in crown (caries) / root (periapical lesion) | – |
| Dentist vs AI (50 X-rays) | sensitivity · PPV · kappa, dentist + AI | – |

## 🚀 How to run

**Colab (recommended, free GPU):** open [Google Colab](https://colab.research.google.com) →
*File → Upload notebook* → choose `notebooks/dental_charting_colab.ipynb` (or click *Open in Colab*
above) → `Runtime → Change runtime type → T4 GPU` → run the cells in order. The notebook carries
the project code, so nothing else needs uploading. Training takes about 1.5 hours; models and
results are saved to your Google Drive, and training resumes automatically if Colab disconnects.

**Locally:**

```bash
pip install -r requirements.txt
python -m pytest tests -q          # end-to-end test on fake X-rays, CPU, about 5 minutes
```

## 📁 Repository structure

```
├── notebooks/dental_charting_colab.ipynb   # the whole study, step by step
├── dentchart/
│   ├── fdi.py         # FDI numbering, DENTEX label mapping, left-right mirroring
│   ├── data.py        # unpack DENTEX, read annotations, splits, PyTorch dataset
│   ├── model.py       # Faster R-CNN (ResNet-50 FPN v2, COCO-pretrained)
│   ├── train.py       # training loop, resumable checkpoints
│   ├── evaluate.py    # COCO AP and dental-chart metrics
│   ├── explain.py     # Grad-CAM for detections + crown/root plausibility check
│   ├── agreement.py   # blinded dentist study: charting tool and agreement statistics
│   └── viz.py         # X-rays with boxes, dental chart, heat-maps, study plots
├── scripts/make_synthetic_dentex.py        # fake DENTEX download for testing
└── tests/                                  # end-to-end and oracle tests
```

## ⚠️ Limitations

- DENTEX boxes outline the whole abnormal tooth, not the lesion itself, so the explanation check
  uses crown and root halves rather than exact lesion outlines.
- In a two-stage detector the classifier only sees features inside the box, so the heat-map shows
  *where inside the tooth* the model looks, not whether it looks at the right tooth.
- The study has one dentist and 50 X-rays, so the estimates have wide confidence intervals. DENTEX
  labels were also made by dentists and can contain errors.
- Retrospective data and no external test site yet. **Research use only; not a medical device.**

## 📄 Citation

If you use this code, please also cite DENTEX:

```
Hamamci, I. E., Er, S., Simsar, E., et al. (2023). DENTEX: An Abnormal Tooth Detection with
Dental Enumeration and Diagnosis Benchmark for Panoramic X-rays. arXiv:2305.19112.
```

## 👤 Author

**Prabhu Raj Singh**, prosthodontist (MDS) and health informatician (MSc Health Informatics,
Karolinska Institutet & Stockholm University) ·
[GitHub](https://github.com/prabhursingh) ·
[LinkedIn](https://www.linkedin.com/in/prabhu-raj-singh-9a926a20) ·
[Website](https://prabhursingh.github.io)

<sub>Code: MIT licence. Figures that show DENTEX X-rays are shared under CC BY-NC-SA 4.0.</sub>
