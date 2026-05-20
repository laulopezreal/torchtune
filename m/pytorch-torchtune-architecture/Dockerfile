FROM python:3.12-slim
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1

SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y \
    git build-essential pkg-config libssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Clone repository at the PR's BASE SHA (pre-merge state)
RUN git clone https://github.com/pytorch/torchtune /workspace/repo && cd /workspace/repo && \
    BASE=213f38605ff0b7b1e20f85a9e032710be04c82c9 && \
    cur_branch="$(git rev-parse --abbrev-ref HEAD)" && \
    git reset --hard "$BASE" && \
    git for-each-ref --format='%(refname:short)' refs/heads | while read -r b; do \
      [ "$b" = "$cur_branch" ] || git branch -D "$b"; \
    done && \
    git remote | xargs -r -n1 git remote remove && \
    rm -f .git/config.lock && \
    git reflog expire --expire=now --all && \
    git gc --prune=now --aggressive

WORKDIR /workspace/repo

# Best-effort install. Code QA SFT tasks don't need to run, only to be readable.
RUN pip install -e . 2>/dev/null || \
    pip install -r requirements.txt 2>/dev/null || \
    pip install -r requirements-dev.txt 2>/dev/null || \
    true
RUN pip install pytest 2>/dev/null || true

WORKDIR /workspace
CMD ["bash"]
