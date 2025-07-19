# syntax=docker/dockerfile:1.6
FROM nvidia/cuda:12.6.2-cudnn-devel-ubuntu24.04 AS base

# ──────────────────────────────────────────────────────────────
# 1. System packages  (C tool-chain + git + Python 3.12)
# ──────────────────────────────────────────────────────────────
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev python3-pip \
        git ca-certificates \
        build-essential clang llvm \
        cuda-compiler-12-6 \
        cuda-nvcc-12-6 \
        cuda-cudart-dev-12-6 \
        cuda-command-line-tools-12-6 \
        cuda-driver-dev-12-6 \
    && rm -rf /var/lib/apt/lists/*

ENV CUDA_HOME=/usr/local/cuda-12.6
ENV PATH="$CUDA_HOME/bin:${PATH}"
ENV LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH}"

# simple alias so "python" is always there
RUN ln -s /usr/bin/python3.12 /usr/local/bin/python

# ──────────────────────────────────────────────────────────────
# 2. uv installer + clone code snapshot
# ──────────────────────────────────────────────────────────────
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
ENV UV_PYTHON=python3.12

WORKDIR /src
ARG REPO_URL=https://github.com/damek/cs336-assignment1-basics.git
ARG REPO_REF=main
RUN git clone --branch "$REPO_REF" --depth 1 "$REPO_URL" repo && git -C repo config --system --add safe.directory /src/repo && chmod -R a+rwx /src/repo

# ──────────────────────────────────────────────────────────────
# 3. Install ALL deps into repo-local venv (.venv/)
# ──────────────────────────────────────────────────────────────
WORKDIR /src/repo
RUN uv sync --locked --no-editable && \
    uv pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu126 --upgrade && \
    uv pip install triton --upgrade && \
    chmod -R a+rX .venv

# ──────────────────────────────────────────────────────────────
# 4. Runtime env  (venv + cache dirs)
# ──────────────────────────────────────────────────────────────
ENV VIRTUAL_ENV=/src/repo/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
ENV WANDB_DIR=/src/repo/cs336_basics/wandb \
    HF_HOME=/src/repo/hf_cache \
    TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor

WORKDIR /src/repo

# ──────────────────────────────────────────────────────────────
# 5. Default command  (override in interactive mode)
# ──────────────────────────────────────────────────────────────
CMD ["python", "-m", "cs336_basics.train"]