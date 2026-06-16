# Sensor-fusion stack: the unified environment that runs ALL light modalities in
# one process for decision-level (late) fusion — LiDAR 3D (spconv) + camera 2D
# (ultralytics) + radar (scikit-learn) — associated in BEV with the repo's
# ByteTrack3D tracker, plus the LiDAR->camera geometric projection.
#
# Feasible because every container is standardised on numpy<2, so the vision and
# lidar stacks no longer conflict. (The official heavy-fusion model BEVFusion still
# lives in the separate `mmdet3d` container — different torch/mmcv.)
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    YOLO_CONFIG_DIR=/tmp/Ultralytics

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libgomp1 ffmpeg git build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Explicit GPU torch (cu128 wheels self-contained; only the driver matters).
RUN pip install --no-cache-dir torch==2.10.0 torchvision==0.25.0 \
        --index-url https://download.pytorch.org/whl/cu128

# Base 3D + viz + nuscenes stack from the Lidar requirements (brings spconv,
# numpy 1.26, opencv-headless 4.11, matplotlib, pandas, nuscenes-devkit).
COPY Lidar/requirements.txt /tmp/lidar_req.txt
RUN printf 'torch==2.10.0\ntorchvision==0.25.0\n' > /tmp/constraints.txt \
    && grep -vE '^(torch|torchvision)([=<> ]|$)' /tmp/lidar_req.txt > /tmp/lidar_req.docker.txt \
    && pip install --no-cache-dir -c /tmp/constraints.txt -r /tmp/lidar_req.docker.txt

# Add the camera detector (ultralytics) + radar association + 3D viz extras +
# the YOLOP lane detector's runtime deps (YOLOP is fetched via torch.hub at first
# use; these are the third-party imports its model code needs).
RUN pip install --no-cache-dir -c /tmp/constraints.txt \
        ultralytics==8.4.14 scikit-learn pyquaternion plotly \
        efficientnet_pytorch prefetch_generator yacs tqdm

# Only the repo root (config + tracking/) on the path by default. Lidar/Vision/Radar
# each expose a top-level `src` package, so they CANNOT all sit on PYTHONPATH at
# once (they'd shadow each other). The fusion orchestration code must import each
# modality with a controlled, one-at-a-time sys.path insertion (or run them as
# subprocesses). See docker/README.md.
ENV PYTHONPATH=/workspace:/workspace/Fusion

CMD ["bash"]
