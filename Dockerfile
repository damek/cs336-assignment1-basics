# syntax=docker/dockerfile:1.6
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# ---- Python 3.11 + uv (all from Ubuntu repos) ----
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3.11-distutils python3-pip \
        curl git && \
    ln -sf python3.11 /usr/bin/python3 && \
    pip3 install --no-cache-dir uv && \
    rm -rf /var/lib/apt/lists/*

# uv installs into ~/.local/bin
ENV PATH="/root/.local/bin:${PATH}"

# ---- lock-correct dependency install ----
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv pip install --system --no-cache-dir   # honours uv.lock

# ---- copy source & run ----
COPY . .
CMD ["uv", "run", "python", "train.py"]
