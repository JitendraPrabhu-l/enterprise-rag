port_mappings = {
  ingestion     = { container_port = 30001, host_port = 18001 }
  retrieval     = { container_port = 30002, host_port = 18002 }
  generation    = { container_port = 30003, host_port = 18003 }
  eval          = { container_port = 30004, host_port = 18004 }
  grafana       = { container_port = 30300, host_port = 13000 }
  phoenix       = { container_port = 30606, host_port = 16006 }
  minio_api     = { container_port = 30900, host_port = 19000 }
  minio_console = { container_port = 30901, host_port = 19001 }
}
