FROM ghcr.io/astral-sh/uv:debian AS base

# install system dependencies (as root)
RUN apt-get update && \
    apt install -yyq --no-install-recommends \
    build-essential \
    portaudio19-dev

# create a non-root user and prepare app dir
RUN mkdir -p /app && \
    useradd -ms /bin/bash nonroot

WORKDIR /app
ENV PYTHONUNBUFFERED=1
ENV PATH=$PATH:/home/nonroot/.local/bin
ENV PYTHONPATH="${PYTHONPATH}:${PWD}"
ENV LLVM_CONFIG=/usr/bin/llvm-config-14

# Copy only dependency metadata and sync script first to maximize cache reuse
# Changing source code later won't invalidate the layer that installs deps.
COPY pyproject.toml ./
COPY scripts/* ./scripts/
RUN chmod +x ./scripts/uv-full-sync.sh

FROM base AS deps

# Install python + project dependencies as nonroot
USER nonroot
WORKDIR /app
RUN mkdir -p /home/nonroot/.local && \
    ./scripts/uv-full-sync.sh

# Final image: copy the rest of the repository and set correct ownership
FROM deps AS dist
USER root
WORKDIR /app
COPY --chown=nonroot:nonroot . .
USER nonroot
ENV PYTHONPATH="${PYTHONPATH}:${PWD}"
CMD ["bash"]
