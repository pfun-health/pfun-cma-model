FROM python:3.10-alpine

# Specify the health check command
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 CMD curl --fail http://localhost:8000/ || exit 1

RUN apk update && apk upgrade && \
    apk add --no-cache \
        python3 \
        make \
        bash \
        curl \
        git \
    && \
    export PATH=/root/.local/bin:/usr/local/bin:/bin:/usr/bin:$PATH && \
    python3 -m ensurepip && \
    rm -r /usr/lib/python*/ensurepip && \
    pip3 install --upgrade pip setuptools && \
    if [ ! -e /usr/bin/pip ]; then ln -s pip3 /usr/bin/pip ; fi && \
    if [[ ! -e /usr/bin/python ]]; then ln -sf /usr/bin/python3 /usr/bin/python; fi && \
    rm -r /root/.cache

CMD ["/bin/bash", "-c", "cd /root/Git/pfun-cma-model && make deploy"]