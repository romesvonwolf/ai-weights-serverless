# AI weight painting — RunPod Serverless worker (UniRig + SkinTokens).
#
# Predicts per-vertex skin weights for our EXISTING skeleton on a GPU, then the
# handler NN-transfers them onto our full-res mesh. Separate from the CPU
# Blender bone-heat worker (runpod/blender-weights).
#
# GPU: needs a CUDA GPU with >= 16 GB VRAM. L4 / A5000 / 4090 / L40S all work.
#
# Build is GitHub-repo-backed on RunPod (tag = git short sha).
#
# BASE: runpod/pytorch (same tag our proven hy-motion worker builds on). Using
# RunPod's own image means python3.11 + CUDA + git + system libs are already set
# up and the layer is cached on RunPod's builder — no apt/deadsnakes risk.
#
# WHY THE INSTALLS ARE SPLIT INTO SMALL PINNED GROUPS (this is the important
# part): an earlier nvidia/cuda build failed on a single giant
# `pip install -r requirements.txt`. That UniRig requirements file pulls open3d,
# whose dependency tree is enormous (dash/plotly/jupyter/scikit-learn/
# matplotlib/...). Resolving it all at once is memory-heavy and the RunPod
# builder OOM-killed pip (it resolves fine on a high-RAM box). hy-motion avoids
# this by pre-installing pinned deps in small steps and running the repo
# requirements filtered + non-fatal. We do the same.
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
# include 12.0 (Blackwell / RTX PRO 6000) — RunPod serverless serves sm_120 GPUs.
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;12.0+PTX"
ENV UNIRIG_DIR=/opt/UniRig
ENV SKINTOKENS_DIR=/opt/SkinTokens
ENV HF_HOME=/opt/hf-cache

# System libs that open3d / bpy / pyrender load at runtime (headless GL etc.).
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl \
    libgl1 libglib2.0-0 libxrender1 libxi6 libxkbcommon0 libsm6 libxext6 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# torch 2.7.0 (cu128). RunPod serverless serves Blackwell GPUs (RTX PRO 6000,
# sm_120); torch <=2.4/cu12.1 has no sm_120 kernels, so the worker's CUDA
# fitness check fails and it exits unhealthy (jobs never dispatch). cu128 torch
# 2.7 ships Blackwell kernels — the same choice our working hy-motion worker
# made. torch's pip wheel bundles its own CUDA 12.8 libs, so the cu12.4 base is
# irrelevant. numpy MUST stay 1.26.x for UniRig.
RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128 \
    && pip install --no-cache-dir numpy==1.26.4

# GPU extension wheels (prebuilt; nothing compiles). pyg + flash-attn matched to
# torch 2.7 / cu128.
RUN pip install --no-cache-dir torch_scatter torch_cluster \
    -f https://data.pyg.org/whl/torch-2.7.0+cu128.html
RUN pip install --no-cache-dir \
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post1/flash_attn-2.8.3.post1+cu12torch2.7cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
# spconv IS a required UniRig dependency — its run.py (skin + skeleton inference)
# imports it at module load, so without it inference crashes and produces no
# output. PyPI's highest CUDA build is spconv-cu120; it bundles its own CUDA 12.0
# runtime and ships kernels for sm_86/sm_89, i.e. our A5000 / L4 / RTX 3090 pool,
# so it runs fine next to the cu128 torch wheel. (spconv-cu120 has NO sm_120
# kernels — an additional reason this endpoint is pinned to Ampere/Ada GPUs.)
RUN pip install --no-cache-dir spconv-cu120

# Blender as a python module (UniRig's extractor + our bpy helpers import bpy).
RUN pip install --no-cache-dir bpy==4.2.0

# --- UniRig's runtime deps, pre-installed as small pinned GROUPS (low peak
# memory; each group is a tiny resolve). Versions mirror UniRig requirements.txt.
RUN pip install --no-cache-dir transformers==4.51.3 huggingface_hub safetensors accelerate
RUN pip install --no-cache-dir pytorch_lightning lightning timm einops omegaconf python-box addict
RUN pip install --no-cache-dir trimesh fast-simplification psutil runpod scipy
# open3d with --no-deps. UniRig only uses open3d for HEADLESS mesh geometry
# (io / geometry / utility), which needs just numpy (already installed). open3d's
# declared deps pull a web/viz tree (dash -> flask -> blinker), and the base
# image has a distutils-installed `blinker` that pip refuses to uninstall to
# upgrade ("Cannot uninstall blinker ... distutils installed project") — that was
# failing every build. --no-deps sidesteps the whole tree and saves ~1 GB.
# Non-fatal: a hiccup here must never block the build (verified at runtime).
RUN pip install --no-cache-dir --no-deps open3d==0.18.0 || echo "WARN: open3d install skipped"
# Peripheral (rendering / logging) — not needed for skinning inference, so don't
# let them fail the build.
RUN pip install --no-cache-dir pyrender wandb || echo "WARN: pyrender/wandb optional deps failed"

# --- UniRig (default model) ---
RUN git clone --depth 1 https://github.com/VAST-AI-Research/UniRig.git ${UNIRIG_DIR}
# Catch-all for anything in UniRig's requirements we didn't pin above. Filter
# out everything we already control (torch/vision/numpy/flash-attn/bpy/open3d)
# so the resolver can't churn them, and keep it NON-FATAL — the runtime deps are
# already installed above; this only backfills extras. Re-pin numpy after.
RUN cd ${UNIRIG_DIR} \
    && grep -ivE '^(torch|torchvision|numpy|flash[-_]attn|bpy|open3d)' requirements.txt > /tmp/unirig_extra.txt || true \
    && echo "=== UniRig backfill reqs ===" && cat /tmp/unirig_extra.txt \
    && pip install --no-cache-dir -r /tmp/unirig_extra.txt || echo "WARN: some UniRig backfill deps failed (core deps already installed)" \
    && pip install --no-cache-dir numpy==1.26.4 \
    && rm -f /tmp/unirig_extra.txt

# NOTE: the skinning checkpoint is intentionally NOT baked in — UniRig's run.py
# downloads it on first request into HF_HOME (the worker's container disk).

# --- SkinTokens (experimental, --model skintokens) ---
RUN git clone --depth 1 https://github.com/VAST-AI-Research/SkinTokens.git ${SKINTOKENS_DIR} || \
    echo "WARN: SkinTokens clone failed; unirig still available"
RUN if [ -f ${SKINTOKENS_DIR}/requirements.txt ]; then \
      cd ${SKINTOKENS_DIR} \
      && grep -ivE '^(torch|torchvision|numpy|flash[-_]attn|bpy|open3d)' requirements.txt > /tmp/st_extra.txt || true \
      && pip install --no-cache-dir -r /tmp/st_extra.txt || echo "WARN: SkinTokens deps failed; unirig still available"; \
    fi

# torch.load weights_only fix (DETERMINISTIC — primary). torch>=2.6 defaults
# torch.load(weights_only=True) and Lightning forwards it explicitly, so UniRig's
# checkpoints (which embed a python-box Box) fail to unpickle ("Unsupported
# global: box.box.Box"). We inject `import torch_compat` as the FIRST line of
# run.py / demo.py so the override runs in-process, in normal program flow,
# before the trainer loads the .ckpt. This replaces the earlier sitecustomize.py
# approach, which this base image's interpreter did NOT auto-import at startup.
COPY torch_compat.py ${UNIRIG_DIR}/torch_compat.py
RUN sed -i '1i import torch_compat' ${UNIRIG_DIR}/run.py \
    && echo "=== run.py head after patch ===" && head -3 ${UNIRIG_DIR}/run.py
RUN if [ -f ${SKINTOKENS_DIR}/demo.py ]; then \
      cp ${UNIRIG_DIR}/torch_compat.py ${SKINTOKENS_DIR}/torch_compat.py \
      && sed -i '1i import torch_compat' ${SKINTOKENS_DIR}/demo.py \
      && echo "patched SkinTokens demo.py"; \
    fi

# Harmless backstop only (kept in case any other code path calls torch.load
# without going through run.py); the entrypoint injection above is the real fix.
COPY sitecustomize.py /usr/local/lib/python3.11/dist-packages/sitecustomize.py

# --- worker ---
WORKDIR /app
COPY handler.py /app/handler.py
COPY blender_build_input.py /app/blender_build_input.py
COPY blender_read_skin.py /app/blender_read_skin.py

CMD ["python", "-u", "/app/handler.py"]
