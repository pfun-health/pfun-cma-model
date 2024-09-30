FROM python:3.10-slim-buster

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    sudo \
    g++ \
    gfortran \
    pkg-config \
    uvicorn \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /app && \
    useradd -ms /bin/bash nonroot

RUN chown -R nonroot:nonroot /app

USER nonroot

## TODO: setup nonroot user with sudoers

WORKDIR /app

COPY . .

RUN pip install --user --upgrade pip setuptools meson && \
    PATH=$PATH:$HOME/.local/bin pip install --user . && \
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc" && \
    export PATH="$HOME/.local/bin:$PATH"

EXPOSE 8000
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH="${PYTHONPATH}:${PWD}"
ENV LLVM_CONFIG=/usr/bin/llvm-config-14
