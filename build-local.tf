provider "local" {}

variable "z38s_ip" {
  description = "IP address of z38s (your desktop)"
  default     = "192.168.1.64"
}

variable "d2bd_ip" {
  description = "IP address of d2bd (your local build server)"
  default     = "192.168.1.158"
}

resource "local_file" "Dockerfile" {
  content = <<-EOF
    # Use an official Python runtime as a parent image
    FROM python:3.10-slim as builder

    # Install any needed packages specified in requirements.txt
    COPY requirements.txt /requirements.txt
    RUN pip install --no-cache-dir -r /requirements.txt

    # Use a smaller base image
    FROM python:3.10-alpine

    # Copy installed packages from builder
    COPY --from=builder /usr/local /usr/local

    # Add your application
    COPY . /app
    WORKDIR /app

    # Run your application
    CMD ["python", "your_app.py"]
  EOF

  filename = "${path.module}/Dockerfile"
}

resource "local_file" "docker_compose" {
  content = <<-EOF
    version: '3.9'
    services:
      model:
        image: rocapp/pfun-cma-model:latest
        container_name: pfun-cma-model
        build:
          context: .
          dockerfile: Dockerfile
        ports:
          - "8003:8001"
          - "8002:8002"
        expose: [8002,8003]
        restart: unless-stopped
  EOF

  filename = "${path.module}/docker-compose.yaml"
}

resource "local_file" "Helm_Chart" {
  content = <<-EOF
    apiVersion: v2
    name: pfun-cma-model
    description: Helm chart for PFun CMA Model
    version: 0.1.0
  EOF

  filename = "${path.module}/Chart.yaml"
}

resource "null_resource" "build_project" {
  depends_on = [local_file.Dockerfile, local_file.docker_compose, local_file.Helm_Chart]

  provisioner "local-exec" {
    command = "docker build -t rocapp/pfun-cma-model:latest ."
  }

  triggers = {
    always_run = "${timestamp()}"
  }
}

resource "null_resource" "test_local_z38s" {
  depends_on = [null_resource.build_project]

  provisioner "remote-exec" {
    inline = [
      "cd /home/robertc/Git/pfun-cma-model",
      "docker-compose up -d",
      "poetry run pytest tests/"
    ]
    connection {
      type        = "ssh"
      host        = var.d2bd_ip
      user        = "your_username"
      private_key = file("~/.ssh/id_rsa")
    }
  }

  triggers = {
    always_run = "${timestamp()}"
  }
}

resource "null_resource" "test_local_d2bd" {
  depends_on = [null_resource.build_project]

  provisioner "remote-exec" {
    inline = [
      "cd /path/to/your/project",
      "docker-compose up -d",
      "pytest tests/"
    ]

    connection {
      type        = "ssh"
      host        = var.d2bd_ip
      user        = "your_username"
      private_key = file("~/.ssh/id_rsa")
    }
  }

  triggers = {
    always_run = "${timestamp()}"
  }
}
