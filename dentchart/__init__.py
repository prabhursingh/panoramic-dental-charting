"""dentchart - dental charting on panoramic X-rays (DENTEX).

Modules
    fdi        FDI tooth numbering and DENTEX label helpers
    data       unpack DENTEX, read annotations, splits, PyTorch dataset
    model      Faster R-CNN (ResNet-50 FPN, COCO-pretrained)
    train      training loop with resumable checkpoints
    evaluate   COCO metrics and dental-chart metrics
    explain    Grad-CAM and the anatomical plausibility check
    agreement  dentist-versus-AI study
    viz        figures
"""

__version__ = "0.1.0"
