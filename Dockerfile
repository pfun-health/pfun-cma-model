# Install uv
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Create a nonroot user
RUN useradd -m -u 1000 nonroot

# Change the working directory to the `app` directory
WORKDIR /app

# Install dependencies
RUN --mount=type=cache,target=/home/nonroot/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=packages,target=packages \
    uv sync --locked --no-install-project \
        --all-extras \
        --group perplexity \
        --group gradio

# Copy the project into the image
COPY --chown=nonroot . /app

# Change ownership of /app to nonroot user
RUN chown -R nonroot:nonroot /app

# Sync the project
RUN --mount=type=cache,target=/home/nonroot/.cache/uv \
    uv sync --locked \
        --all-extras \
        --group perplexity \
        --group gradio

# Switch to nonroot user
USER nonroot
