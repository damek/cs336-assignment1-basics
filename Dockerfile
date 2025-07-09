# syntax=docker/dockerfile:1.6
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VER=3.11.8        # bump when you upgrade

# ------------------------------------------------------------
# 1.  Build & install CPython 3.11
# ------------------------------------------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential curl ca-certificates git \
        zlib1g-dev libbz2-dev libssl-dev libreadline-dev \
        libsqlite3-dev libncurses5-dev libncursesw5-dev \
        libffi-dev liblzma-dev libgdbm-dev libnss3-dev uuid-dev && \
    curl -fsSL https://www.python.org/ftp/python/${PYTHON_VER}/Python-${PYTHON_VER}.tgz | tar xz && \
    cd Python-${PYTHON_VER} && \
    ./configure --enable-optimizations --with-ensurepip=install && \
    make -j"$(nproc)" && make altinstall && \
    cd .. && rm -rf Python-${PYTHON_VER} && \
    rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------
# 2.  uv itself
# ------------------------------------------------------------
RUN pip3.11 install --no-cache-dir uv

# uv installs its shims into ~/.local/bin
ENV PATH="/root/.local/bin:${PATH}"

# tell uv to install into the system interpreter
ENV UV_PROJECT_ENVIRONMENT=system

WORKDIR /app

# ------------------------------------------------------------
# 3.  Dependency layer – only reruns when these two files change
# ------------------------------------------------------------
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-cache            # exact, reproducible deps

# ------------------------------------------------------------
# 4.  Rest of your source
# ------------------------------------------------------------
COPY . .

CMD ["uv", "run", "python", "train.py"]
