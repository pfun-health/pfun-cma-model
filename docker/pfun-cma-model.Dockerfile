FROM python:3.11-slim-buster

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    g++ \
    gfortran \
    pkg-config \
    meson \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# create a non-root user
# and set the app root directory
RUN mkdir -p /app && \
    useradd -ms /bin/bash nonroot


# set the app root directory
WORKDIR /app

# copy as root
COPY . .
# setup permissions (as root, for the non-root user)
RUN chown -R nonroot:nonroot /app
RUN chmod -R ug+rw /app

USER nonroot
RUN echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"

USER nonroot
RUN mkdir -p /app/minpack/_build
RUN export PATH="/home/nonroot/.local/bin:$PATH" && \
    pip install --user --upgrade pip && \
    ./install.sh

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH="${PYTHONPATH}:${PWD}"
ENV LLVM_CONFIG=/usr/bin/llvm-config-14
