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
USER nonroot
WORKDIR /app
ENV PYTHONUNBUFFERED=1
ENV PATH=$PATH:/home/nonroot/.local/bin
ENV PYTHONPATH="${PYTHONPATH}:${PWD}"
ENV LLVM_CONFIG=/usr/bin/llvm-config-14
# install python + dependencies
RUN uv sync --project=pfun_gradio --no-cache && \
    uv build --project=pfun_gradio --no-cache

FROM deps AS dist
# final stage (entrypoint)
USER nonroot
WORKDIR /app
# overridden in compose
# Run the application.
CMD ["/app/.venv/bin/fastapi", "run", "pfun_gradio/main.py", "--port", "7860", "--host", "0.0.0.0"]