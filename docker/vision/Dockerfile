# Vision stack: GPU torch + ultralytics (YOLO26, RT-DETR) + Streamlit app deps.
# Base image already ships a CUDA-enabled torch, so we strip the +cpu torch pin
# from Vision/requirements.txt and install everything else on top.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# OpenCV / matplotlib / video runtime libs
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libgomp1 ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install the GPU torch the host uses (cu128 wheels are self-contained: they bundle
# CUDA/cuDNN, so the base image's CUDA version is irrelevant — only the driver
# matters). This overrides the base image torch and stops pip from later pulling a
# CPU torch from PyPI to satisfy a transitive requirement.
RUN pip install --no-cache-dir torch==2.10.0 torchvision==0.25.0 \
        --index-url https://download.pytorch.org/whl/cu128

# Install the remaining deps, EXCEPT torch/torchvision, numpy, and the GUI
# opencv-python. The host pins numpy==2.4.2 + opencv-python==4.13 (both need
# numpy>=2) alongside nuscenes-devkit 1.2.0 (needs numpy<2) — a real conflict the
# host only tolerates with a non-strict resolver. Here we standardise on numpy<2
# (1.26.x) + opencv-python-headless==4.11, and a constraints file pins torch so no
# dependency can swap in a CPU build.
COPY Vision/requirements.txt /tmp/req.txt
RUN printf 'torch==2.10.0\ntorchvision==0.25.0\n' > /tmp/constraints.txt \
    && grep -vE '^(torch|torchvision|numpy|opencv-python)([=<> ]|$)' /tmp/req.txt > /tmp/req.docker.txt \
    && pip install --no-cache-dir "numpy<2" \
    && pip install --no-cache-dir -c /tmp/constraints.txt -r /tmp/req.docker.txt

# Repo + Vision package on the path (code itself is bind-mounted at runtime).
ENV PYTHONPATH=/workspace:/workspace/Vision

CMD ["bash"]
