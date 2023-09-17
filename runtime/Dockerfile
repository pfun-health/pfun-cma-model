#syntax=docker/dockerfile:1.4

# Use a lightweight base image
FROM python:3.10-alpine

# Install dependencies
COPY requirements-docker.txt /app/
COPY requirements.txt /app/
RUN apk add py3-scipy && \
    pip install --upgrade \
	--platform manylinux2014_x86_64 \
	--target /usr/local/lib/python3.10/site-packages \
	--implementation cp \
	--python-version 3.10 \
	--only-binary=:all: \
	--no-cache-dir \
    -r /app/requirements-docker.txt

RUN apk add mlocate && updatedb && \
    echo '# link local executables (e.g., Chalice)' >> $HOME/.bashrc && \
    echo 'export PATH=/usr/local/bin:$HOME/.local/bin:$PATH' >> $HOME/.bashrc && \
    echo 'export CHALICE_EXE=$(locate /bin/chalice | head -1)' >> $HOME/.bashrc

# Copy the application
COPY . /app
WORKDIR /app

ENV CHALICE_PORT=${CHALICE_PORT:-8001}

# Expose the port the app runs on
EXPOSE ${CHALICE_PORT}

# Command to run the application
CMD ["/bin/sh", "-c", "${CHALICE_EXE}", "local", "--port", "${CHALICE_PORT}", "--host", "0.0.0.0"]
