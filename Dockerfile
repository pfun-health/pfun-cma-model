FROM alpine:3.7

WORKDIR /app
COPY requirements-dev.txt ./
COPY requirements.txt ./

RUN \
    apk add --no-cache --virtual .build-deps gcc python3-dev musl-dev && \
    python3 -m pip install \
    -r requirements-dev.txt \
    -r requirements.txt \
    --no-cache-dir && \
    python3 -m pip install \
    --no-cache-dir \
    --upgrade pip && \
    apk --purge del .build-deps

COPY . .
CMD [ "/bin/sh"]