# Lidar + Radar + Fusion stack: GPU torch + spconv (sparse conv) + numpy 1.26.
# Runs the in-repo clean-room detectors (PointPillars / SECOND / CenterPoint),
# the Radar module, and the Fusion projection scripts. These do NOT depend on
# mmdet3d — for the official model zoo use the `mmdet3d` service instead.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libgomp1 ffmpeg git build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install the GPU torch the host uses (cu128 wheels bundle CUDA/cuDNN; only the
# driver matters). Done explicitly so pip can't later pull a CPU torch from PyPI.
RUN pip install --no-cache-dir torch==2.10.0 torchvision==0.25.0 \
        --index-url https://download.pytorch.org/whl/cu128

# Install Lidar deps EXCEPT torch/torchvision; a constraints file pins torch so no
# dependency swaps in a CPU build. spconv-cu120 wheels are self-contained.
COPY Lidar/requirements.txt /tmp/req.txt
RUN printf 'torch==2.10.0\ntorchvision==0.25.0\n' > /tmp/constraints.txt \
    && grep -vE '^(torch|torchvision)([=<> ]|$)' /tmp/req.txt > /tmp/req.docker.txt \
    && pip install --no-cache-dir -c /tmp/constraints.txt -r /tmp/req.docker.txt

# Only Lidar on the path: Lidar/src and Radar/src are both top-level `src`
# namespace packages and would shadow each other. Radar has its own container.
ENV PYTHONPATH=/workspace:/workspace/Lidar

CMD ["bash"]
