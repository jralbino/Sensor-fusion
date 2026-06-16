# Official OpenMMLab 3D model zoo: BEVFusion / CMT / CenterPoint / DSVT / VoxelNeXt
# with their REAL pretrained checkpoints, plus NuScenes data preparation
# (tools/create_data.py). This stack pins an older torch (2.1) because mmcv/mmdet3d
# do not yet build against torch 2.10 — exactly why it lives in its own container.
#
# A "devel" base image is required to compile mmcv / mmdet3d CUDA ops.
# Build is heavy (~15-30 min, compiles ops) — best built on the RTX 3060Ti server.
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    FORCE_CUDA=1 \
    TORCH_CUDA_ARCH_LIST="8.6"

# 8.6 = Ampere (RTX 3050 / 3060Ti / 3090). Adjust if running on other GPUs.

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libgomp1 ffmpeg git ninja-build build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# numpy<2 is required by mmdet3d 1.4 / nuscenes-devkit.
RUN pip install --no-cache-dir "numpy<2"

# OpenMMLab stack via mim (resolves the correct prebuilt mmcv wheel for torch2.1/cu121).
RUN pip install --no-cache-dir -U openmim \
    && mim install "mmengine>=0.10.0" \
    && mim install "mmcv>=2.0.0rc4,<2.2.0" \
    && mim install "mmdet>=3.0.0,<3.3.0" \
    && mim install "mmdet3d>=1.4.0,<1.5.0"

RUN pip install --no-cache-dir nuscenes-devkit==1.2.0 spconv-cu120==2.3.6

ENV PYTHONPATH=/workspace:/workspace/Lidar:/workspace/Fusion

CMD ["bash"]
