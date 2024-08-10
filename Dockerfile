FROM python:3.10-slim-buster

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /app && \
useradd -ms /bin/bash nonroot

RUN chown -R nonroot:nonroot /app

USER nonroot

WORKDIR /app

COPY . .

RUN pip install --user --upgrade pip setuptools && \
    pip install --user . && \
    pip install --user uvicorn && \
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc" && \
    export PATH="$HOME/.local/bin:$PATH"

EXPOSE 8000
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH="${PYTHONPATH}:${PWD}"

ENTRYPOINT ["/usr/bin/env", "/bin/bash", "-c", "python", "-m", "uvicorn", "pfun_cma_model.main:app", "--host", "0.0.0.0", "--port", "8000"]

CMD ["/usr/bin/env", "/bin/bash", "-c", "python", "-m", "uvicorn", "pfun_cma_model.main:app", "--host", "0.0.0.0", "--port", "8000"]
