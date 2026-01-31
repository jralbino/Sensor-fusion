# -*- coding: utf-8 -*-
"""
Model definitions for the Vision component.
"""

OBJECT_DETECTORS = {
    "YOLOv11-L": {"type": "yolo", "key": "yolo11l"},
    "YOLOv11-X": {"type": "yolo", "key": "yolo11x"},
    "RT-DETR-L": {"type": "rtdetr", "key": "rtdetr_l"},
    "RT-DETR-BDD": {"type": "rtdetr", "key": "rtdetr_bdd"},
    "RT-DETR-People": {"type": "rtdetr", "key": "rtdetr_people"},
}

LANE_DETECTORS = {
    "YOLOP": "YOLOP (Panoptic)",
    "UFLD": "UFLD",
    "PolyLaneNet": "PolyLaneNet",
    "DeepLabv3": "DeepLabV3",
    "SegFormer": "SegFormer",
}
