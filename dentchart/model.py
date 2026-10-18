"""Faster R-CNN detector (torchvision) adapted to panoramic X-rays."""

import torch
from torchvision.models.detection import (FasterRCNN_ResNet50_FPN_V2_Weights,
                                          fasterrcnn_resnet50_fpn_v2)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def build_model(num_classes, pretrained=True, min_size=800, max_size=1333, detections_per_img=100):
    """Faster R-CNN with a ResNet-50 FPN backbone.

    Starts from COCO-pretrained weights (transfer learning, as in the MRI project) and
    replaces the last layer so it predicts our classes: 32 teeth or 4 diagnoses
    (+ background). Images are resized so the long side is at most `max_size` pixels;
    a typical OPG (about 2900 x 1300) becomes about 1333 x 600.
    """
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1 if pretrained else None
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, weights_backbone=None, min_size=min_size,
                                       max_size=max_size, box_detections_per_img=detections_per_img)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def load_weights(model, path, device="cpu"):
    """Load a checkpoint written by train.train_detector (best.pt or last.pt)."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    return model
