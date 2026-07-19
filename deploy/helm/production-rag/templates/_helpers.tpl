{{/*
Common labels applied to every resource this chart creates.
*/}}
{{- define "production-rag.labels" -}}
app.kubernetes.io/part-of: production-rag
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Service type — ADR-019: values-kind.yaml sets .Values.service.type to
NodePort for the local kind-cluster path; every other deployment target
stays ClusterIP. Each Service template additionally emits its own
`nodePort:` field (guarded by an `if eq .Values.service.type "NodePort"`)
since that value differs per service.
*/}}
{{- define "production-rag.serviceType" -}}
type: {{ .Values.service.type }}
{{- end }}

{{/*
Shared env vars every application service needs (ADR-018: the Kubernetes
equivalent of docker-compose.yml's x-service-env anchor).
*/}}
{{- define "production-rag.commonEnv" -}}
- name: GROQ_API_KEY
  valueFrom:
    secretKeyRef:
      name: {{ .Release.Name }}-secrets
      key: groq-api-key
- name: GENERATION_MODEL
  value: {{ .Values.env.generationModel | quote }}
- name: VISION_MODEL
  value: {{ .Values.env.visionModel | quote }}
- name: UTILITY_MODEL
  value: {{ .Values.env.utilityModel | quote }}
- name: QDRANT_URL
  value: "http://{{ .Release.Name }}-qdrant:{{ .Values.qdrant.port }}"
- name: OPENSEARCH_URL
  value: "http://{{ .Release.Name }}-opensearch:{{ .Values.opensearch.port }}"
- name: NEO4J_URI
  value: "bolt://{{ .Release.Name }}-neo4j:{{ .Values.neo4j.boltPort }}"
- name: NEO4J_USER
  value: {{ .Values.env.neo4jUser | quote }}
- name: NEO4J_PASSWORD
  valueFrom:
    secretKeyRef:
      name: {{ .Release.Name }}-secrets
      key: neo4j-password
- name: REDIS_URL
  value: "redis://{{ .Release.Name }}-redis:{{ .Values.redis.port }}/0"
- name: RATE_LIMIT_PER_MINUTE
  value: {{ .Values.env.rateLimitPerMinute | quote }}
- name: MINIO_ENDPOINT_URL
  value: "http://{{ .Release.Name }}-minio:{{ .Values.minio.apiPort }}"
- name: MINIO_ACCESS_KEY
  valueFrom:
    secretKeyRef:
      name: {{ .Release.Name }}-secrets
      key: minio-root-user
- name: MINIO_SECRET_KEY
  valueFrom:
    secretKeyRef:
      name: {{ .Release.Name }}-secrets
      key: minio-root-password
- name: MINIO_BUCKET
  value: {{ .Values.env.minioBucket | quote }}
- name: OTEL_EXPORTER_OTLP_ENDPOINT
  value: "http://{{ .Release.Name }}-otel-collector:{{ .Values.otelCollector.httpPort }}"
- name: OTEL_TRACES_SAMPLE_RATE
  value: {{ .Values.env.otelTracesSampleRate | quote }}
- name: ENVIRONMENT
  value: {{ .Values.env.environment | quote }}
- name: LOG_LEVEL
  value: {{ .Values.env.logLevel | quote }}
{{- end }}
