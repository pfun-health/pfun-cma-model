# syntax = docker/dockerfile:1.2

FROM python:3.10 as base
LABEL Author, Robbie Capps


# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1
# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

ENV PATH=/root/.local/bin:/root/.local:$PATH

#: reset cache
RUN rm -f /etc/apt/apt.conf.d/docker-clean

#: install apt dependencies (e.g., git)
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && \
    apt-get install -yqq --no-install-recommends \
    git jq libnss3-tools && \
    rm -rf /var/lib/apt/lists/*

# install mkcert for dev ssl
RUN curl -JLO "https://dl.filippo.io/mkcert/latest?for=linux/amd64" && \
    chmod +x mkcert-v*-linux-amd64 && \
    cp mkcert-v*-linux-amd64 /usr/local/bin/mkcert

# pip deps
COPY ./requirements.txt ./requirements.txt
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install \
    --user --upgrade \
    --no-cache-dir \
    -r requirements.txt

# copy app filfes
COPY ./app ./app

# generate certfiles
RUN mkcert localhost 127.0.0.1 0.0.0.0 ::1

# run
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--ssl-keyfile", "localhost+3-key.pem", "--ssl-certfile", "localhost+3.pem"]