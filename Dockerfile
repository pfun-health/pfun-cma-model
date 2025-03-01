FROM python:3.10-slim-buster

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    g++ \
    gfortran \
    pkg-config \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /app && \
    useradd -ms /bin/bash nonroot

RUN chown -R nonroot:nonroot /app

USER nonroot

WORKDIR /app

COPY . .

RUN echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"

RUN export PATH="/home/nonroot/.local/bin:$PATH" && \
    pip install --user --upgrade pip && \
    pip install --user .

EXPOSE 8000
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH="${PYTHONPATH}:${PWD}"
ENV LLVM_CONFIG=/usr/bin/llvm-config-14
