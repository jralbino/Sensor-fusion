"""
MMDetection3D SECOND Wrapper

The MMDet3D "PointPillars" checkpoint uses a SECOND backbone + SECONDFPN
internally (hence the name ``hv_pointpillars_secfpn_sbn-all``). This wrapper
exposes the same model under the ``mmdet3d_second`` name, since the backbone
architecture is genuinely SECOND.

Same weights, same architecture, different name for clarity.
"""

from src.detectors.mmdet3d_pointpillars import MMDet3DPointPillars


class MMDet3DSECOND(MMDet3DPointPillars):
    """MMDet3D SECOND -- uses PointPillars checkpoint (SECOND backbone + SECONDFPN).

    This is a thin alias: the MMDet3D PointPillars checkpoint already uses
    the SECOND backbone architecture with dense 2D convolutions, so the
    weights and forward pass are identical.
    """
    pass
