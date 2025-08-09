FROM ghcr.io/astral-sh/uv:debian as base

# create a non-root user
# and set the app root directory
RUN mkdir -p /app && \
    useradd -ms /bin/bash nonroot

# set the app root directory
WORKDIR /app
# copy as root
COPY --chown=nonroot:nonroot . .
# ensure permissions for nonroot
RUN chown nonroot:nonroot .

FROM base as deps

# install python + dependencies
USER nonroot
WORKDIR /app
ENV PYTHONUNBUFFERED=1
ENV PATH=$PATH:/home/nonroot/.local/bin
ENV PYTHONPATH="${PYTHONPATH}:${PWD}"
ENV LLVM_CONFIG=/usr/bin/llvm-config-14
RUN \
    uv venv && \
    uv add fastapi --extra standard && \
    uv sync && \
    uv build


FROM deps as dist
USER nonroot
WORKDIR /app
# overridden in compose
CMD ["bash"]
