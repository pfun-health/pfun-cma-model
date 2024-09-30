FROM python:3.10-slim-buster

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    sudo \
    g++ \
    gfortran \
    meson \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /app && \
    useradd -ms /bin/bash nonroot

RUN chown -R nonroot:nonroot /app

USER nonroot

## TODO: setup nonroot user with sudoers

WORKDIR /app

COPY . .

RUN pip install --user --upgrade pip setuptools && \
    pip install --user . && \
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc" && \
    export PATH="$HOME/.local/bin:$PATH"

EXPOSE 8000
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH="${PYTHONPATH}:${PWD}"

RUN python ./scripts/build_minpack.py

CMD ["/usr/bin/env", "/bin/bash", "-c", "python", "-m", "pfun_cma_model.cli", "launch"]
