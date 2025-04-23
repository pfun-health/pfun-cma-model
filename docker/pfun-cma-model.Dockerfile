FROM python:3.11-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    g++ \
    gfortran \
    pkg-config \
    python3-venv \
    python-is-python3 \
    python3-pip python3-setuptools \
    python3-wheel ninja-build \
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

USER nonroot
RUN echo 'export PATH="/home/nonroot/.local/bin:$PATH"' >> "/home/nonroot/.bashrc"

USER nonroot
WORKDIR /app
# install meson and build minpack (from the context root directory)
RUN pip3 install --user meson
RUN bash -c 'export PATH=/home/nonroot/.local/bin:$PATH; rm -rf ./_build && ./install.sh'
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH="${PYTHONPATH}:${PWD}"
ENV LLVM_CONFIG=/usr/bin/llvm-config-14
