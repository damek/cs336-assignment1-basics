# syntax=docker/dockerfile:1.6
FROM nvidia/cuda:12.6.2-cudnn-runtime-ubuntu22.04

# ── 1.  System Python 3.11 + pip ──────────────────────────────────────────────
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip && \
    rm -rf /var/lib/apt/lists/*

# make "python" resolve to python3.11
RUN ln -s /usr/bin/python3.11 /usr/local/bin/python

# ── 2.  Fast Python toolchain (uv) ────────────────────────────────────────────
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
ENV UV_SYSTEM_PYTHON=1 UV_PYTHON=python3.11

# ── 3.  Install ALL deps *system-wide* as root (world-readable) ───────────────
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-editable && \
    chmod -R a+rX /usr/local/lib/python3.11          

# Extra CUDA wheels (torch + friends) — pick cu121 index for CUDA 12.x
RUN pip3 install --no-cache-dir --upgrade \
        --index-url https://download.pytorch.org/whl/cu121 \
        torch torchvision torchaudio && \
    pip3 install --no-cache-dir einops regex wandb tqdm

# ── 4.  Non-root user just for the workdir (no writes needed later) ───────────
ARG UID=1000
RUN groupadd -g $UID appuser && useradd -u $UID -g $UID -m appuser
USER appuser

# project code (read-only is fine)
COPY --chown=appuser:appuser cs336_basics ./cs336_basics

# ── 5.  Runtime ───────────────────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1
EXPOSE 8888
CMD ["python", "-m", "cs336_basics.train"]
