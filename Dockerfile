FROM --platform=linux/amd64 public.ecr.aws/lambda/python:3.10

WORKDIR /app
COPY requirements-dev.txt ./
COPY requirements.txt ./

RUN yum install -y git python3 && \
    yum clean all && \
    python3 -m pip install \
        -r requirements-dev.txt \
        -r requirements.txt \
        --no-cache-dir

COPY ./chalicelib ${LAMBDA_RUNTIME_DIR}
COPY ./app.py ${LAMBDA_TASK_ROOT}
CMD ["app.app"]