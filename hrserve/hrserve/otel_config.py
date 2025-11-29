"""
OpenTelemetry configuration for Music Models servers.

Configure via environment variables:
- OTEL_EXPORTER_OTLP_ENDPOINT: OTLP endpoint (e.g., http://localhost:4318)
- OTEL_SERVICE_NAME: Service name override (default: server-specific)
"""

import os
import logging

# OpenTelemetry is optional
try:
    from opentelemetry import trace, metrics
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
    from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter
    from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
    from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
    from opentelemetry._logs import set_logger_provider
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False


def setup_otel(service_name: str, service_version: str = "1.0.0"):
    """
    Setup OpenTelemetry tracing, metrics, and logging.

    Args:
        service_name: Name of the service (e.g., "orpheus-server", "deepseek-server")
        service_version: Version of the service

    Returns:
        (tracer, meter) tuple if OTEL_EXPORTER_OTLP_ENDPOINT is set, else (None, None)
    """
    if not OTEL_AVAILABLE:
        print("⚠️  OpenTelemetry not installed. Telemetry disabled.")
        return None, None

    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")

    if not endpoint:
        print("⚠️  OTEL_EXPORTER_OTLP_ENDPOINT not set. OpenTelemetry disabled.")
        return None, None

    # Allow service name override
    service_name = os.getenv("OTEL_SERVICE_NAME", service_name)

    print(f"📊 Configuring OpenTelemetry:")
    print(f"   Service: {service_name} v{service_version}")
    print(f"   Endpoint: {endpoint}")

    try:
        # Create resource with service metadata
        resource = Resource.create({
            "service.name": service_name,
            "service.version": service_version,
        })

        # Setup tracing
        trace_provider = TracerProvider(resource=resource)
        trace_exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
        trace_provider.add_span_processor(BatchSpanProcessor(trace_exporter))
        trace.set_tracer_provider(trace_provider)

        # Setup metrics
        metric_exporter = OTLPMetricExporter(endpoint=endpoint, insecure=True)
        metric_reader = PeriodicExportingMetricReader(metric_exporter, export_interval_millis=60000)
        meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
        metrics.set_meter_provider(meter_provider)

        # Setup logging
        logger_provider = LoggerProvider(resource=resource)
        log_exporter = OTLPLogExporter(endpoint=endpoint, insecure=True)
        logger_provider.add_log_record_processor(BatchLogRecordProcessor(log_exporter))
        set_logger_provider(logger_provider)

        # Attach OTEL handler to root logger
        handler = LoggingHandler(logger_provider=logger_provider)
        logging.getLogger().addHandler(handler)

        print("✓ OpenTelemetry configured (traces + metrics + logs)")

        tracer = trace.get_tracer(__name__)
        meter = metrics.get_meter(__name__)

        return tracer, meter
    except Exception as e:
        print(f"⚠️  Failed to configure OpenTelemetry: {e}")
        print("   Server will continue without telemetry")
        return None, None
