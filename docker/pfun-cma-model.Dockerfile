FROM python:3.11-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    g++ \
    gfortran \
    pkg-config \
    python3-venv \
    python-is-python3 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# create a non-root user
# and set the app root directory
RUN mkdir -p /app && \
    useradd -ms /bin/bash nonroot


# set the app root directory
WORKDIR /app

# copy as root
COPY --chown=nonroot:nonroot . .

USER nonroot
RUN echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"

USER nonroot
RUN mkdir -p /app/minpack/_build

USER nonroot
RUN bash -c "./install.sh"
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH="${PYTHONPATH}:${PWD}"
ENV LLVM_CONFIG=/usr/bin/llvm-config-14
