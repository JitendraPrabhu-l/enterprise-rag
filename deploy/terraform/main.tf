# ADR-019: Provisions a local `kind` (Kubernetes-in-Docker) cluster —
# genuinely zero-cost, requires only a working Docker daemon. Terraform's
# job ends at "a working cluster + kubeconfig exist"; deploying the Helm
# chart (deploy/helm/production-rag) onto it is a separate, explicit step
# (see README.md in this directory).

terraform {
  required_version = ">= 1.5"

  required_providers {
    kind = {
      source  = "tehcyx/kind"
      version = "~> 0.7"
    }
    local = {
      source  = "hashicorp/local"
      version = "~> 2.5"
    }
  }
}

provider "kind" {}

resource "kind_cluster" "production_rag" {
  name           = var.cluster_name
  wait_for_ready = true

  kind_config {
    kind        = "Cluster"
    api_version = "kind.x-k8s.io/v1alpha4"

    node {
      role = "control-plane"

      # Host port mappings mirror docker-compose.yml's own port mappings,
      # so a service is reachable at the same localhost:<port> regardless
      # of whether the stack is run via Compose or via this Terraform+Helm
      # path.
      dynamic "extra_port_mappings" {
        for_each = var.port_mappings
        content {
          container_port = extra_port_mappings.value.container_port
          host_port      = extra_port_mappings.value.host_port
          protocol       = "TCP"
        }
      }
    }
  }
}

resource "local_file" "kubeconfig" {
  content         = kind_cluster.production_rag.kubeconfig
  filename        = "${path.module}/kubeconfig"
  file_permission = "0600"
}
