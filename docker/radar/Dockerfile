# Radar stack: GPU torch + scikit-learn (CFAR+DBSCAN classical, RadarPillars,
# RadarCenterPoint) + NuScenes radar loading. Light stack, isolated per the
# "one container per sensor" layout.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libgomp1 ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Explicit GPU torch (cu128 wheels are self-contained; only the driver matters).
RUN pip install --no-cache-dir torch==2.10.0 torchvision==0.25.0 \
        --index-url https://download.pytorch.org/whl/cu128

# Radar deps, EXCEPT torch (pinned via constraints) and the GUI opencv-python
# (needs numpy>=2, which conflicts with nuscenes-devkit). Use headless + numpy<2.
COPY Radar/requirements.txt /tmp/req.txt
RUN printf 'torch==2.10.0\ntorchvision==0.25.0\n' > /tmp/constraints.txt \
    && grep -vE '^(torch|torchvision|numpy|opencv-python)([=<> ]|$)' /tmp/req.txt > /tmp/req.docker.txt \
    && pip install --no-cache-dir "numpy<2" "opencv-python-headless<4.12" \
    && pip install --no-cache-dir -c /tmp/constraints.txt -r /tmp/req.docker.txt

ENV PYTHONPATH=/workspace:/workspace/Radar

CMD ["bash"]
