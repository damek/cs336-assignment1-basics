# syntax=docker/dockerfile:1.6
FROM nvidia/cuda:12.6.2-cudnn-runtime-ubuntu22.04

# ────────────────────────────────────────────────────────────────────────────
# 1. System packages
# ---------------------------------------------------------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip \
        git ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# handy alias so "python" always exists (but we'll override PATH later)
RUN ln -s /usr/bin/python3.11 /usr/local/bin/python

# ────────────────────────────────────────────────────────────────────────────
# 2. Fast installer (uv) + clone your repo
# ---------------------------------------------------------------------------
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
ENV UV_PYTHON=python3.11       

WORKDIR /src
ARG REPO_URL=https://github.com/damek/cs336-assignment1-basics.git
ARG REPO_REF=main
RUN git clone --branch "$REPO_REF" --depth 1 "$REPO_URL" repo

# ────────────────────────────────────────────────────────────────────────────
# 3. Install ALL dependencies into a project-local virtual-env
#    (uv creates .venv/ inside the repo)
# ---------------------------------------------------------------------------
WORKDIR /src/repo
RUN uv sync --locked --no-editable && \
    chmod -R a+rX .venv                      

# ────────────────────────────────────────────────────────────────────────────
# 4. Make the repo tree writable, then drop privileges
# ---------------------------------------------------------------------------
ARG UID=1000
RUN chown -R $UID:$UID /src/repo
RUN groupadd -g $UID appuser && useradd -u $UID -g $UID -m appuser
USER appuser

# ────────────────────────────────────────────────────────────────────────────
# 5. Activate the venv for **every** command / subprocess
# ---------------------------------------------------------------------------
ENV VIRTUAL_ENV=/src/repo/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Optional but handy: keep WandB logs and HF downloads inside the repo too
ENV WANDB_DIR=/src/repo/cs336_basics/wandb \
    HF_HOME=/src/repo/hf_cache

WORKDIR /src/repo           

# ────────────────────────────────────────────────────────────────────────────
# 6. Default entrypoint
# ---------------------------------------------------------------------------
CMD ["python", "-m", "cs336_basics.train"]