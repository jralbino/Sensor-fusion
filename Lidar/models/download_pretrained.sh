#!/bin/bash
# Download pretrained MMDet3D checkpoints for NuScenes

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# PointPillars SECFPN
PP_NAME="hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d_20210826_225857-f19d00a3.pth"
PP_URL="https://download.openmmlab.com/mmdetection3d/v1.0.0_models/pointpillars/hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d/${PP_NAME}"

# CenterPoint Pillar SECFPN (circle NMS)
CP_NAME="centerpoint_02pillar_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220811_031844-191a3822.pth"
CP_URL="https://download.openmmlab.com/mmdetection3d/v1.0.0_models/centerpoint/centerpoint_02pillar_second_secfpn_circlenms_4x8_cyclic_20e_nus/${CP_NAME}"

download() {
    local url="$1"
    local dest="$2"
    if [ -f "$dest" ]; then
        echo "Already exists: $(basename "$dest")"
        return
    fi
    echo "Downloading $(basename "$dest")..."
    wget -q --show-progress -O "$dest" "$url"
    if [ $? -eq 0 ]; then
        echo "  Saved: $dest"
    else
        echo "  FAILED: $url"
        rm -f "$dest"
        return 1
    fi
}

echo "=== Downloading pretrained MMDet3D checkpoints ==="
echo ""

download "$PP_URL" "${SCRIPT_DIR}/${PP_NAME}"
download "$CP_URL" "${SCRIPT_DIR}/${CP_NAME}"

echo ""
echo "Done. Checkpoints saved to: ${SCRIPT_DIR}/"
ls -lh "${SCRIPT_DIR}"/*.pth 2>/dev/null
