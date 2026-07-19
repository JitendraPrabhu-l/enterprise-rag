variable "cluster_name" {
  description = "Name of the kind cluster."
  type        = string
  default     = "production-rag"
}

variable "port_mappings" {
  description = <<-EOT
    Host <-> container port mappings for the kind control-plane node,
    mirroring docker-compose.yml's own host port mappings (ADR-019) so a
    service is reachable at the same localhost:<port> regardless of which
    orchestration path (Compose or Terraform+Helm) stood the stack up.
  EOT
  type = map(object({
    container_port = number
    host_port      = number
  }))
  default = {
    ingestion     = { container_port = 30001, host_port = 8001 }
    retrieval     = { container_port = 30002, host_port = 8002 }
    generation    = { container_port = 30003, host_port = 8003 }
    eval          = { container_port = 30004, host_port = 8004 }
    grafana       = { container_port = 30300, host_port = 3000 }
    phoenix       = { container_port = 30606, host_port = 6006 }
    minio_api     = { container_port = 30900, host_port = 9000 }
    minio_console = { container_port = 30901, host_port = 9001 }
  }
}
