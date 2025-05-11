FROM python:3.11-bookworm-slim as base

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    g++ \
    gfortran \
    pkg-config \
    python3-venv \
    python-is-python3 \
    python3-pip \
    pipx \
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
# ensure permissions for nonroot
RUN chown -R nonroot:nonroot /app/

FROM base as deps

# install python + dependencies
USER nonroot
WORKDIR /app
ENV PYTHONUNBUFFERED=1
ENV PATH=$PATH:/home/nonroot/.local/bin
ENV PYTHONPATH="${PYTHONPATH}:${PWD}"
ENV LLVM_CONFIG=/usr/bin/llvm-config-14
RUN \
    pipx install poetry && \
    poetry install


FROM deps as test

# run pytest in poetry virtual env
RUN \
    poetry run pytest


FROM deps as dist

# overridden in compose
CMD ["bash"]
