output "kubeconfig_path" {
  description = "Path to the kubeconfig for the provisioned kind cluster — point kubectl/helm at this."
  value       = local_file.kubeconfig.filename
}

output "cluster_endpoint" {
  description = "Kubernetes API server endpoint for the provisioned cluster."
  value       = kind_cluster.production_rag.endpoint
}

output "cluster_name" {
  value = kind_cluster.production_rag.name
}
