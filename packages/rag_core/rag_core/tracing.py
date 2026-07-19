"""OpenTelemetry setup shared by all services (ADR: 100% error / sampled success trace).

Every service calls `configure_tracing(settings)` once at startup and uses
`get_tracer(__name__)` thereafter — this keeps exporter/sampler config in one
place instead of duplicated per service.
"""

from __future__ import annotations

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased
from opentelemetry.trace import Tracer

from rag_core.config import BaseServiceSettings


def configure_tracing(settings: BaseServiceSettings) -> None:
    resource = Resource.create({SERVICE_NAME: settings.service_name})
    sampler = ParentBased(TraceIdRatioBased(settings.otel_traces_sample_rate))
    provider = TracerProvider(resource=resource, sampler=sampler)
    exporter = OTLPSpanExporter(endpoint=f"{settings.otel_exporter_otlp_endpoint}/v1/traces")
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)


def get_tracer(name: str) -> Tracer:
    return trace.get_tracer(name)
