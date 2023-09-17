from hashicorp/terraform:1.3.10 as tf0

from python:3.10.10

COPY --from=tf0 /bin/terraform /bin/

RUN pip install terraform-local

ARG AWS_ENDPOINT_URL="${AWS_ENDPOINT_URL}"
RUN echo "AWS_ENDPOINT_URL=${AWS_ENDPOINT_URL}" > /etc/environment
RUN echo "S3_HOSTNAME=${AWS_ENDPOINT_URL}}" >> /etc/environment
RUN echo "AWS_ENDPOINT_URL=${AWS_ENDPOINT_URL}"