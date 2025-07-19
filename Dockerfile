# syntax=docker/dockerfile:1.6
FROM nvidia/cuda:12.6.2-cudnn-devel-ubuntu24.04 AS base

# Combine installs and clean up aggressively
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
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/* \
    && rm -rf /var/tmp/*

ENV CUDA_HOME=/usr/local/cuda-12.6
ENV PATH="$CUDA_HOME/bin:${PATH}"
ENV LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH}"
RUN ln -s /usr/bin/python3.12 /usr/local/bin/python

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
ENV UV_PYTHON=python3.12

WORKDIR /src
ARG REPO_URL=https://github.com/damek/cs336-assignment1-basics.git
ARG REPO_REF=main
RUN git clone --branch "$REPO_REF" --depth 1 "$REPO_URL" repo && \
    git -C repo config --system --add safe.directory /src/repo && \
    chmod -R a+rwx /src/repo

WORKDIR /src/repo

# Install packages and clean up in one layer
RUN uv sync --locked --no-editable && \
    uv pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu126 --upgrade && \
    uv pip install triton --upgrade && \
    chmod -R a+rX .venv && \
    # Clean up caches to save space
    uv cache clean && \
    find /src/repo/.venv -name "*.pyc" -delete && \
    find /src/repo/.venv -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

ENV VIRTUAL_ENV=/src/repo/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
ENV WANDB_DIR=/src/repo/cs336_basics/wandb \
    HF_HOME=/src/repo/hf_cache \
    TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_cache \
    TORCH_HOME=/tmp/torch_cache \
    XDG_CACHE_HOME=/tmp/cache

RUN mkdir -p /tmp/torchinductor_cache /tmp/torch_cache /tmp/cache && \
    chmod -R 777 /tmp/torchinductor_cache /tmp/torch_cache /tmp/cache

CMD ["python", "-m", "cs336_basics.train"]