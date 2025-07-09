# syntax=docker/dockerfile:1.6
    FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04 AS base

    ENV DEBIAN_FRONTEND=noninteractive
    RUN apt-get update && \
        apt-get install -y --no-install-recommends \
            curl build-essential git ca-certificates && \
        rm -rf /var/lib/apt/lists/*
    

    ARG PYTHON_VER=3.11.8
    RUN curl -sSL https://www.python.org/ftp/python/${PYTHON_VER}/Python-${PYTHON_VER}.tgz | tar xz && \
        cd Python-${PYTHON_VER} && \
        ./configure --enable-optimizations --with-ensurepip=install && \
        make -j"$(nproc)" && make altinstall && \
        cd .. && rm -rf Python-${PYTHON_VER}
    ENV PATH="/usr/local/bin:$PATH"
    
    RUN python3.11 -m pip install --no-cache-dir uv==0.1.41
    
    WORKDIR /train_models
    
    COPY pyproject.toml uv.lock* ./
    
    RUN uv pip install --system --prod
    
    COPY . .
    
    RUN python3.11 -m pip install --no-deps -e .
    
    CMD ["bash"]
    