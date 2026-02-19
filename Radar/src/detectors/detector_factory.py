# -*- coding: utf-8 -*-
"""
Factory for radar detectors.
"""
from __future__ import annotations

from typing import Any


def get_radar_detector(model_type: str, **kwargs) -> Any:
    """Create a radar detector instance by type name.

    Args:
        model_type: One of 'cfar_dbscan', 'radar_pillars', 'radar_centerpoint'.
        **kwargs: Forwarded to the detector constructor.

    Returns:
        Detector instance.
    """
    model_type = model_type.lower()

    if model_type == 'cfar_dbscan':
        from Radar.src.detectors.cfar_dbscan import CFARDBSCANDetector
        return CFARDBSCANDetector(**kwargs)
    elif model_type == 'radar_pillars':
        from Radar.src.detectors.radar_pillars import RadarPillars
        return RadarPillars(**kwargs)
    elif model_type == 'radar_centerpoint':
        from Radar.src.detectors.radar_centerpoint import RadarCenterPoint
        return RadarCenterPoint(**kwargs)
    else:
        raise ValueError(
            f"Unknown radar detector: {model_type!r}. "
            f"Choose from: cfar_dbscan, radar_pillars, radar_centerpoint"
        )
