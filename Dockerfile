# AI weight painting — RunPod Serverless worker (UniRig + SkinTokens).
#
# Predicts per-vertex skin weights for our EXISTING skeleton on a GPU, then the
# handler NN-transfers them onto our full-res mesh. Separate from the CPU
# Blender bone-heat worker (runpod/blender-weights).
#
# GPU: needs a CUDA GPU with >= 16 GB VRAM (UniRig generation needs ~8 GB; give
# headroom for SkinTokens). L4 / A5000 / 4090 / L40S class all work.
#
# Build is GitHub-repo-backed on RunPod (tag = git short sha), same as the
# blender-weights worker.

# RUNTIME (not devel) base: ~4.5 GB smaller. Nothing here compiles CUDA at build
# time — torch, spconv, torch_scatter/cluster and flash-attn all install from
# prebuilt wheels — so we don't need nvcc/the devel toolkit. Keeping the image
# small matters: the GitHub builder failed on the devel image (out of build disk
# while writing the OCI output tar of a 20 GB+ image).
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0+PTX"
ENV UNIRIG_DIR=/opt/UniRig
ENV SKINTOKENS_DIR=/opt/SkinTokens
ENV HF_HOME=/opt/hf-cache

# --- system + python 3.11 ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common git curl wget build-essential ninja-build \
    libgl1 libglib2.0-0 libxrender1 libxi6 libxkbcommon0 libsm6 libxext6 \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3.11-venv \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# --- torch (cu121) + numpy pin (UniRig requires numpy 1.26.4) ---
RUN pip install --upgrade pip setuptools wheel \
    && pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121 \
    && pip install numpy==1.26.4

# --- sparse-conv + scatter/cluster + flash-attn (UniRig install guide) ---
RUN pip install spconv-cu120
RUN pip install torch_scatter torch_cluster \
    -f https://data.pyg.org/whl/torch-2.3.1+cu121.html --no-cache-dir
# flash-attn: install the PREBUILT wheel (torch2.3 / cu12 / py311; abiFALSE
# matches pip-installed torch). NEVER source-build it here — flash-attn from
# source takes 30-90 min and routinely OOMs CI builders. UniRig's
# requirements.txt also lists flash_attn (unpinned); installing it here first
# means that step finds it already satisfied and won't trigger a source build.
RUN pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu123torch2.3cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"

# --- Blender as a python module (extractor + our bpy helpers import bpy) ---
RUN pip install bpy==4.2.0

# --- UniRig (default model) ---
RUN git clone --depth 1 https://github.com/VAST-AI-Research/UniRig.git ${UNIRIG_DIR}
# requirements.txt pulls flash_attn (already satisfied by the wheel above) +
# bpy==4.2 + transformers/lightning/open3d/pyrender/etc. Re-pin numpy AFTER, as
# some of those deps will otherwise upgrade numpy past 1.26.x and break UniRig.
RUN cd ${UNIRIG_DIR} && pip install -r requirements.txt psutil runpod scipy \
    && pip install numpy==1.26.4

# NOTE: the skinning checkpoint is intentionally NOT baked into the image — it
# adds ~2 GB to the image + build disk (which is what was overflowing the
# builder) and needs HF access at build time. UniRig's run.py downloads it
# automatically on the first request into HF_HOME (the worker's container disk).

# --- SkinTokens (experimental, --model skintokens) ---
RUN git clone --depth 1 https://github.com/VAST-AI-Research/SkinTokens.git ${SKINTOKENS_DIR} || \
    echo "WARN: SkinTokens clone failed; unirig still available"
RUN cd ${SKINTOKENS_DIR} && pip install -r requirements.txt || \
    echo "WARN: SkinTokens deps failed; unirig still available"

# --- worker ---
WORKDIR /app
COPY handler.py /app/handler.py
COPY blender_build_input.py /app/blender_build_input.py
COPY blender_read_skin.py /app/blender_read_skin.py

CMD ["python", "-u", "/app/handler.py"]
