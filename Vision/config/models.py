# -*- coding: utf-8 -*-
"""
Model definitions for the Vision component.
"""

OBJECT_DETECTORS = {
    "YOLO26-L": {"type": "yolo", "key": "yolo26l"},
    "YOLO26-X": {"type": "yolo", "key": "yolo26x"},
    "YOLOv11-L": {"type": "yolo", "key": "yolo11l"},
    "YOLOv11-X": {"type": "yolo", "key": "yolo11x"},
    "RT-DETR-X": {"type": "rtdetr", "key": "rtdetr_x"},
    "RT-DETR-L": {"type": "rtdetr", "key": "rtdetr_l"},
    "RT-DETR-BDD": {"type": "rtdetr", "key": "rtdetr_bdd"},
    "RT-DETR-People": {"type": "rtdetr", "key": "rtdetr_people"},
}

LANE_DETECTORS = {
    "YOLOP": "YOLOP (Panoptic)",
    "UFLD": "UFLD (TuSimple)",
    "UFLD (CULane)": "UFLD (CULane)",
    "PolyLaneNet": "PolyLaneNet",
    "DeepLabv3": "DeepLabV3",
    "SegFormer": "SegFormer",
}
