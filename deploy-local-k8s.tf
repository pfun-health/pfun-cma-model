variable "create_helm_repo" {
  description = "Toggle to create Helm repo locally"
  default     = false
}

resource "null_resource" "init_helm_repo" {
  count = var.create_helm_repo ? 1 : 0

  provisioner "local-exec" {
    command = "helm repo add local-repo http://localhost:8879/charts"

    environment = {
      KUBECONFIG = var.kubeconfig_path
    }
  }

  triggers = {
    always_run = "${timestamp()}"
  }
}

resource "null_resource" "test_local_z38s" {
  # Your local testing steps for z38s
  provisioner "local-exec" {
    command = <<EOL
      # Your local testing commands for z38s
    EOL

    environment = {
      KUBECONFIG = var.kubeconfig_path
    }
  }

  triggers = {
    always_run = "${timestamp()}"
  }
}

resource "null_resource" "test_local_d2bd" {
  # Your local testing steps for d2bd
  provisioner "local-exec" {
    command = <<EOL
      # Your local testing commands for d2bd
    EOL

    environment = {
      KUBECONFIG = var.kubeconfig_path
    }
  }

  triggers = {
    always_run = "${timestamp()}"
  }
}

resource "null_resource" "deploy_to_local" {
  depends_on = [null_resource.test_local_z38s, null_resource.test_local_d2bd]

  provisioner "local-exec" {
    command = <<EOL
      # Package the Helm chart
      helm package ./path/to/helm/chart

      # Upload the Helm chart to your local chart repository
      helm push ./path/to/helm/chart/package.tgz local-repo

      # Update the Helm repo
      helm repo update

      # Deploy the Helm chart to the local cluster
      helm upgrade --install pfun-cma-model local-repo/pfun-cma-model --namespace your-namespace
    EOL

    environment = {
      KUBECONFIG = var.kubeconfig_path
    }
  }

  triggers = {
    always_run = "${timestamp()}"
  }
}

variable "kubeconfig_path" {
  description = "Path to the kubeconfig file"
  default     = "~/.kube/config"
}
