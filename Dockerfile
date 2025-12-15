FROM ghcr.io/astral-sh/uv:debian AS base

# install system dependencies (as root)

RUN apt-get update && \
    apt install -yyq --no-install-recommends \
    build-essential \
    portaudio19-dev

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

FROM base AS deps

# install python + dependencies
# NOTE: specify extras, groups, e.g., '--extra standard --group perplexity'
USER nonroot
WORKDIR /app
ENV PYTHONUNBUFFERED=1
ENV PATH=$PATH:/home/nonroot/.local/bin
ENV PYTHONPATH="${PYTHONPATH}:${PWD}"
ENV LLVM_CONFIG=/usr/bin/llvm-config-14
RUN \
    uv venv && \
    uv add fastapi --extra standard && \
    uv sync --all-extras --group perplexity --group gradio && \
    uv build


FROM deps AS dist

USER nonroot
WORKDIR /app
# overridden in compose
CMD ["bash"]
