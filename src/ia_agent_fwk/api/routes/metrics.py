"""Prometheus-compatible /metrics endpoint."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from starlette.responses import Response

from ia_agent_fwk.observability.metrics import get_metrics_collector

router = APIRouter(tags=["observability"])

_CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"


def _format_prometheus(snapshot: dict[str, Any]) -> str:
    """Convert MetricsCollector snapshot to Prometheus text exposition format."""
    lines: list[str] = []

    # --- Counters ---
    counters: dict[str, Any] = snapshot.get("counters", {})
    for name, buckets in sorted(counters.items()):
        lines.append(f"# HELP {name} Counter metric.")
        lines.append(f"# TYPE {name} counter")
        for label_key, value in sorted(buckets.items()):
            if label_key:
                # label_key is "k1=v1,k2=v2" -> convert to {k1="v1",k2="v2"}
                parts = label_key.split(",")
                label_str = ",".join(f'{k}="{v}"' for k, v in (p.split("=", 1) for p in parts))
                lines.append(f"{name}{{{label_str}}} {value}")
            else:
                lines.append(f"{name} {value}")

    # --- Histograms ---
    histograms: dict[str, Any] = snapshot.get("histograms", {})
    # Group labeled histograms by base name for proper HELP/TYPE headers
    seen_histogram_names: set[str] = set()
    for composite_name, stats in sorted(histograms.items()):
        count = stats.get("count", 0)
        total = stats.get("sum", 0.0)

        # Parse out embedded labels: "metric{k=v,...}" -> ("metric", "{k=v,...}")
        if "{" in composite_name and composite_name.endswith("}"):
            brace_idx = composite_name.index("{")
            base_name = composite_name[:brace_idx]
            label_key = composite_name[brace_idx + 1 : -1]
            # Convert "k1=v1,k2=v2" -> '{k1="v1",k2="v2"}'
            parts = label_key.split(",")
            label_str = ",".join(f'{k}="{v}"' for k, v in (p.split("=", 1) for p in parts))
            label_block = f"{{{label_str}}}"
        else:
            base_name = composite_name
            label_block = ""

        if base_name not in seen_histogram_names:
            lines.append(f"# HELP {base_name} Histogram metric.")
            lines.append(f"# TYPE {base_name} summary")
            seen_histogram_names.add(base_name)
        lines.append(f"{base_name}_count{label_block} {count}")
        lines.append(f"{base_name}_sum{label_block} {total}")

    lines.append("")
    return "\n".join(lines)


@router.get("/metrics")
async def prometheus_metrics() -> Response:
    """Expose metrics in Prometheus text exposition format."""
    collector = get_metrics_collector()
    snapshot = collector.snapshot()
    body = _format_prometheus(snapshot)
    return Response(content=body, media_type=_CONTENT_TYPE)
