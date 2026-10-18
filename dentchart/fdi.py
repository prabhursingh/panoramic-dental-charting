"""FDI tooth numbering and DENTEX label helpers.

DENTEX labels every box with three numbers:

    category_id_1 -> quadrant index 0..3   (quadrant 1..4)
    category_id_2 -> tooth index    0..7   (tooth 1..8)
    category_id_3 -> diagnosis index 0..3  (see DIAGNOSES)

so the FDI number of a box is (category_id_1 + 1) * 10 + (category_id_2 + 1).

On a panoramic X-ray (OPG) the patient's right side is shown on the viewer's left.
Quadrant 1 (upper right) is therefore in the top-left of the image and quadrant 4
(lower right) in the bottom-left.
"""

QUADRANTS = (1, 2, 3, 4)

# 11..18, 21..28, 31..38, 41..48 - the 32 permanent teeth.
TEETH = tuple(q * 10 + t for q in QUADRANTS for t in range(1, 9))

# Detector class ids: 0 is background, 1..32 are teeth.
TOOTH_TO_LABEL = {fdi: i + 1 for i, fdi in enumerate(TEETH)}
LABEL_TO_TOOTH = {label: fdi for fdi, label in TOOTH_TO_LABEL.items()}

# DENTEX order of category_id_3. The real order is read from the annotation file
# when it is available (see data.diagnosis_names); this is only the fallback.
DIAGNOSES = ("Impacted", "Caries", "Periapical Lesion", "Deep Caries")

# Short codes used when a dentist charts the X-rays by hand.
CODES = {"Caries": "C", "Deep Caries": "DC", "Periapical Lesion": "PA", "Impacted": "I"}
CODE_TO_DIAGNOSIS = {code: name for name, code in CODES.items()}

# Where the evidence for each diagnosis should be on the tooth (used by the
# explanation check). Impacted teeth have no single expected region.
EXPECTED_REGION = {"Caries": "crown", "Deep Caries": "crown", "Periapical Lesion": "root"}

COLORS = {"Caries": "#f28e2b", "Deep Caries": "#e15759", "Periapical Lesion": "#9c6ade",
          "Impacted": "#4e79a7"}


def fdi_from_dentex(category_id_1, category_id_2):
    """DENTEX quadrant/tooth indices (0-based) -> FDI number, e.g. (3, 7) -> 48."""
    return (int(category_id_1) + 1) * 10 + (int(category_id_2) + 1)


def mirror(fdi):
    """FDI number of the same tooth after flipping the X-ray left-right (e.g. 36 -> 46)."""
    quadrant, tooth = divmod(int(fdi), 10)
    return {1: 2, 2: 1, 3: 4, 4: 3}[quadrant] * 10 + tooth


def is_upper(fdi):
    """True for maxillary teeth (quadrants 1 and 2)."""
    return int(fdi) // 10 in (1, 2)


def chart_rows():
    """Teeth in the order a dentist reads an OPG: patient's right on the viewer's left.

    Returns (upper_row, lower_row), each with 16 FDI numbers.
    """
    upper = [10 + t for t in range(8, 0, -1)] + [20 + t for t in range(1, 9)]
    lower = [40 + t for t in range(8, 0, -1)] + [30 + t for t in range(1, 9)]
    return upper, lower


def canonical_diagnosis(name):
    """Map spelling variants ('periapical lesion', 'Deep caries') to the canonical names."""
    key = " ".join(str(name).replace("_", " ").split()).lower()
    for canonical in DIAGNOSES:
        if canonical.lower() == key:
            return canonical
    return str(name)
