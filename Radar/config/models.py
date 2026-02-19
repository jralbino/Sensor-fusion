# -*- coding: utf-8 -*-
"""
Model definitions for the Radar component.
"""

RADAR_DETECTORS = {
    "CFAR+DBSCAN": {"type": "cfar_dbscan", "key": None},
    "RadarPillars": {"type": "radar_pillars", "key": "radar_pillars"},
    "RadarCenterPoint": {"type": "radar_centerpoint", "key": "radar_centerpoint"},
}
