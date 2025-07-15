# syntax=docker/dockerfile:1.6
# ────────────────────────────────────────────────────────────────────────────────
# CUDA + cuDNN runtime base; minimal but GPU-ready
FROM nvidia/cuda:12.6.2-cudnn-runtime-ubuntu22.04

# 1 ── System packages ───────────────────────────────────────────────────────────
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip git ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# make "python" map to python3.11 (needed by sweep.py subprocess)
RUN ln -s /usr/bin/python3.11 /usr/local/bin/python

# uv – ultra-fast lock-aware installer
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
ENV UV_SYSTEM_PYTHON=1 UV_PYTHON=python3.11

# 2 ── Clone your repo (always latest main) ─────────────────────────────────────
# ARG lets you pin a different branch or commit when building
ARG REPO_URL=https://github.com/damek/cs336-assignment1-basics.git
ARG REPO_REF=main

WORKDIR /src
RUN git clone --branch "$REPO_REF" --depth 1 "$REPO_URL" repo

# 3 ── Install ALL dependencies *system-wide* as root ───────────────────────────
WORKDIR /src/repo
# uv sync installs *everything* in uv.lock (torch, einops, regex, wandb, …)
RUN uv sync --locked --no-editable && \
    chmod -R a+rX /usr/local/lib/python3.11

# Extra: make sure CUDA wheels for torch are present (if not in lock)
# Comment these two lines if torch is already in uv.lock pointing at cu121 wheels
RUN pip3 install --no-cache-dir --upgrade pip wheel setuptools && \
    pip3 install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 \
        torch torchvision torchaudio

# 4 ── Non-root runtime user (Run:AI security) ──────────────────────────────────
ARG UID=1000
RUN groupadd -g $UID appuser && useradd -u $UID -g $UID -m appuser
USER appuser
WORKDIR /app
RUN ln -s /src/repo/cs336_basics ./cs336_basics   

# 5 ── Runtime config ───────────────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1
EXPOSE 8888
CMD ["python", "-m", "cs336_basics.train"]