#!/usr/bin/env python3
"""Generate the Streaming Grafana dashboard JSON.

Covers:
- SSE endpoint streaming
- WebSocket bidirectional communication
- Stream event types (start, heartbeat, complete, error)
- Streaming configuration (heartbeat, ping, max connections)
- Client disconnects and error handling
- LLM-level streaming backpressure

Run:
    python3 scripts/generate_streaming_dashboard.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# --- Constants ---
PROM_DS = {"type": "prometheus", "uid": "PBFA97CFB590B2093"}
LOKI_DS = {"type": "loki", "uid": "loki-datasource"}
DASHBOARD_UID = "ia-agent-fwk-streaming"

_next_id = 0


def _id() -> int:
    global _next_id
    _next_id += 1
    return _next_id


def _grid(col: int, row: int, w: int = 6, h: int = 8) -> dict[str, int]:
    return {"h": h, "w": w, "x": col, "y": row}


def _prom(expr: str, legend: str = "", instant: bool = False) -> dict[str, Any]:
    t: dict[str, Any] = {"datasource": PROM_DS, "expr": expr, "legendFormat": legend}
    if instant:
        t["instant"] = True
        t["range"] = False
    return t


def _loki(expr: str) -> dict[str, Any]:
    return {"datasource": LOKI_DS, "expr": expr, "queryType": "range"}


def stat(title: str, targets: list, pos: dict, color: str = "green", unit: str = "short",
         thresholds: list | None = None) -> dict[str, Any]:
    if thresholds is None:
        thresholds = [{"color": color, "value": None}]
    return {
        "id": _id(), "type": "stat", "title": title, "datasource": PROM_DS,
        "targets": targets, "gridPos": pos,
        "options": {"reduceOptions": {"calcs": ["lastNotNull"]}, "colorMode": "background"},
        "fieldConfig": {"defaults": {"unit": unit, "thresholds": {"mode": "absolute", "steps": thresholds}}},
    }


def ts(title: str, targets: list, pos: dict, unit: str = "short", stack: bool = False) -> dict[str, Any]:
    custom: dict[str, Any] = {"drawStyle": "line", "fillOpacity": 10, "lineWidth": 2}
    if stack:
        custom["stacking"] = {"mode": "normal"}
        custom["fillOpacity"] = 30
    return {
        "id": _id(), "type": "timeseries", "title": title, "datasource": PROM_DS,
        "targets": targets, "gridPos": pos,
        "fieldConfig": {"defaults": {"unit": unit, "custom": custom}},
    }


def gauge(title: str, targets: list, pos: dict, unit: str = "percentunit",
          min_val: int = 0, max_val: int = 1, thresholds: list | None = None) -> dict[str, Any]:
    if thresholds is None:
        thresholds = [
            {"color": "green", "value": None},
            {"color": "yellow", "value": 0.7},
            {"color": "red", "value": 0.9},
        ]
    return {
        "id": _id(), "type": "gauge", "title": title, "datasource": PROM_DS,
        "targets": targets, "gridPos": pos,
        "options": {"reduceOptions": {"calcs": ["lastNotNull"]}},
        "fieldConfig": {"defaults": {"unit": unit, "min": min_val, "max": max_val,
                                     "thresholds": {"mode": "absolute", "steps": thresholds}}},
    }


def pie(title: str, targets: list, pos: dict) -> dict[str, Any]:
    return {
        "id": _id(), "type": "piechart", "title": title, "datasource": PROM_DS,
        "targets": targets, "gridPos": pos,
        "options": {"reduceOptions": {"calcs": ["lastNotNull"]},
                    "legend": {"displayMode": "table", "placement": "right"}, "pieType": "donut"},
    }


def bar(title: str, targets: list, pos: dict, unit: str = "short") -> dict[str, Any]:
    return {
        "id": _id(), "type": "bargauge", "title": title, "datasource": PROM_DS,
        "targets": targets, "gridPos": pos,
        "options": {"reduceOptions": {"calcs": ["lastNotNull"]},
                    "displayMode": "gradient", "orientation": "horizontal"},
        "fieldConfig": {"defaults": {"unit": unit}},
    }


def logs(title: str, targets: list, pos: dict) -> dict[str, Any]:
    return {
        "id": _id(), "type": "logs", "title": title, "datasource": LOKI_DS,
        "targets": targets, "gridPos": pos,
        "options": {"showTime": True, "sortOrder": "Descending",
                    "enableLogDetails": True, "wrapLogMessage": True},
    }


def row(title: str, y: int) -> dict[str, Any]:
    return {
        "id": _id(), "type": "row", "title": title, "collapsed": True,
        "gridPos": {"h": 1, "w": 24, "x": 0, "y": y}, "panels": [],
    }


# --- Build ---
def build() -> dict[str, Any]:
    panels: list[dict[str, Any]] = []
    y = 0

    # ================================================================
    # Row 1: Streaming Overview
    # ================================================================
    r = row("Streaming Overview", y)
    y += 1
    rp: list[dict[str, Any]] = []

    # Stats row
    rp.append(stat("SSE Streams", [_prom("sum(sse_streams_total)", instant=True)],
                    _grid(0, y, 4, 4), "blue"))
    rp.append(stat("SSE Completed", [_prom('sum(sse_streams_completed_total{status="success"})', instant=True)],
                    _grid(4, y, 4, 4), "green"))
    rp.append(stat("SSE Errors", [_prom('sum(sse_streams_completed_total{status="error"})', instant=True)],
                    _grid(8, y, 4, 4), "red",
                    thresholds=[{"color": "green", "value": None}, {"color": "red", "value": 1}]))
    rp.append(stat("WS Connections", [_prom("sum(ws_connections_total)", instant=True)],
                    _grid(12, y, 4, 4), "blue"))
    rp.append(stat("WS Agent Runs", [_prom("sum(ws_agent_executions_total)", instant=True)],
                    _grid(16, y, 4, 4), "purple"))
    rp.append(stat("Client Disconnects", [_prom("sum(streaming_client_disconnects_total)", instant=True)],
                    _grid(20, y, 4, 4), "orange",
                    thresholds=[{"color": "green", "value": None}, {"color": "orange", "value": 1}]))
    y += 4

    # Rate panels
    rp.append(ts("Streaming API Requests (per minute)",
                  [_prom('sum(rate(streaming_api_requests_total{transport="sse"}[1m]))*60', "SSE"),
                   _prom('sum(rate(streaming_api_requests_total{transport="websocket"}[1m]))*60', "WebSocket")],
                  _grid(0, y, 8, 8), unit="reqps", stack=True))

    rp.append(ts("Stream Events by Type (per minute)",
                  [_prom('sum(rate(streaming_events_total{event="start"}[1m]))*60', "Start"),
                   _prom('sum(rate(streaming_events_total{event="heartbeat"}[1m]))*60', "Heartbeat"),
                   _prom('sum(rate(streaming_events_total{event="complete"}[1m]))*60', "Complete"),
                   _prom('sum(rate(streaming_events_total{event="error"}[1m]))*60', "Error")],
                  _grid(8, y, 8, 8), stack=True))

    rp.append(pie("Events Distribution",
                   [_prom("sum(streaming_events_total) by (event)", "{{event}}")],
                   _grid(16, y, 8, 8)))
    y += 8

    r["panels"] = rp
    panels.append(r)

    # ================================================================
    # Row 2: SSE Streaming
    # ================================================================
    r = row("SSE Endpoint Streaming", y)
    y += 1
    rp = []

    rp.append(ts("SSE Streams Rate by Agent (per minute)",
                  [_prom("sum(rate(sse_streams_total[1m])) by (agent_type) * 60", "{{agent_type}}")],
                  _grid(0, y, 8, 8), unit="reqps"))

    rp.append(ts("SSE Stream Duration",
                  [_prom("rate(sse_stream_duration_seconds_sum[5m]) / rate(sse_stream_duration_seconds_count[5m])",
                         "Avg Duration")],
                  _grid(8, y, 8, 8), unit="s"))

    rp.append(ts("SSE Completion Status (per minute)",
                  [_prom('sum(rate(sse_streams_completed_total{status="success"}[1m]))*60', "Success"),
                   _prom('sum(rate(sse_streams_completed_total{status="error"}[1m]))*60', "Error"),
                   _prom('sum(rate(sse_streams_completed_total{status="cancelled"}[1m]))*60', "Cancelled")],
                  _grid(16, y, 8, 8), stack=True))
    y += 8

    # SSE stats
    rp.append(stat("Avg Heartbeats/Stream",
                    [_prom("rate(sse_heartbeats_per_stream_sum[5m]) / rate(sse_heartbeats_per_stream_count[5m])",
                           instant=True)],
                    _grid(0, y, 6, 4), "blue"))
    rp.append(stat("SSE Cancelled",
                    [_prom('sum(sse_streams_completed_total{status="cancelled"})', instant=True)],
                    _grid(6, y, 6, 4), "orange",
                    thresholds=[{"color": "green", "value": None}, {"color": "orange", "value": 1}]))
    rp.append(bar("SSE Streams by Agent Type",
                   [_prom("sum(sse_streams_total) by (agent_type)", "{{agent_type}}")],
                   _grid(12, y, 12, 4)))
    y += 4

    r["panels"] = rp
    panels.append(r)

    # ================================================================
    # Row 3: WebSocket Communication
    # ================================================================
    r = row("WebSocket Bidirectional Communication", y)
    y += 1
    rp = []

    rp.append(ts("WS Connections Rate (per minute)",
                  [_prom("sum(rate(ws_connections_total[1m]))*60", "New Connections"),
                   _prom("sum(rate(ws_connections_rejected_total[1m]))*60", "Rejected")],
                  _grid(0, y, 8, 8), unit="reqps"))

    rp.append(ts("WS Agent Execution Duration",
                  [_prom("rate(ws_agent_execution_duration_seconds_sum[5m]) / rate(ws_agent_execution_duration_seconds_count[5m])",
                         "Avg Duration")],
                  _grid(8, y, 8, 8), unit="s"))

    rp.append(ts("WS Agent Completion Status (per minute)",
                  [_prom('sum(rate(ws_agent_executions_completed_total{status="success"}[1m]))*60', "Success"),
                   _prom('sum(rate(ws_agent_executions_completed_total{status="error"}[1m]))*60', "Error")],
                  _grid(16, y, 8, 8), stack=True))
    y += 8

    # WS stats row
    rp.append(stat("WS Auth Success",
                    [_prom('sum(ws_auth_total{status="success"})', instant=True)],
                    _grid(0, y, 4, 4), "green"))
    rp.append(stat("WS Auth Failures",
                    [_prom('sum(ws_auth_total{status="failure"})', instant=True)],
                    _grid(4, y, 4, 4), "red",
                    thresholds=[{"color": "green", "value": None}, {"color": "red", "value": 1}]))
    rp.append(stat("WS Messages Received",
                    [_prom("sum(ws_messages_received_total)", instant=True)],
                    _grid(8, y, 4, 4), "blue"))
    rp.append(stat("WS Invalid Messages",
                    [_prom("sum(ws_invalid_messages_total)", instant=True)],
                    _grid(12, y, 4, 4), "orange",
                    thresholds=[{"color": "green", "value": None}, {"color": "orange", "value": 1}]))
    rp.append(stat("Avg Messages/Connection",
                    [_prom("rate(ws_messages_per_connection_sum[5m]) / rate(ws_messages_per_connection_count[5m])",
                           instant=True)],
                    _grid(16, y, 4, 4), "purple"))
    rp.append(stat("WS Errors",
                    [_prom("sum(ws_errors_total)", instant=True)],
                    _grid(20, y, 4, 4), "red",
                    thresholds=[{"color": "green", "value": None}, {"color": "red", "value": 1}]))
    y += 4

    r["panels"] = rp
    panels.append(r)

    # ================================================================
    # Row 4: Connection Health & Keep-alive
    # ================================================================
    r = row("Connection Health & Keep-alive", y)
    y += 1
    rp = []

    rp.append(ts("WS Active Connections",
                  [_prom("rate(ws_active_connections_sum[1m]) / rate(ws_active_connections_count[1m])",
                         "Active Connections")],
                  _grid(0, y, 8, 8)))

    rp.append(ts("WS Ping Status (per minute)",
                  [_prom('sum(rate(ws_pings_total{status="success"}[1m]))*60', "Success"),
                   _prom('sum(rate(ws_pings_total{status="failure"}[1m]))*60', "Failure (stale)")],
                  _grid(8, y, 8, 8), stack=True))

    rp.append(ts("Client Disconnects (per minute)",
                  [_prom('sum(rate(streaming_client_disconnects_total{transport="sse"}[1m]))*60', "SSE"),
                   _prom('sum(rate(streaming_client_disconnects_total{transport="websocket"}[1m]))*60', "WebSocket")],
                  _grid(16, y, 8, 8), stack=True))
    y += 8

    # Connection stats
    rp.append(stat("WS Connection Duration (avg)",
                    [_prom("rate(ws_connection_duration_seconds_sum[5m]) / rate(ws_connection_duration_seconds_count[5m])",
                           instant=True)],
                    _grid(0, y, 6, 4), "blue", unit="s"))
    rp.append(stat("WS Rejected (max conn)",
                    [_prom('sum(ws_connections_rejected_total{reason="max_connections"})', instant=True)],
                    _grid(6, y, 6, 4), "red",
                    thresholds=[{"color": "green", "value": None}, {"color": "red", "value": 1}]))
    rp.append(stat("Total Pings Sent",
                    [_prom("sum(ws_pings_total)", instant=True)],
                    _grid(12, y, 6, 4), "blue"))
    rp.append(stat("Failed Pings (stale)",
                    [_prom('sum(ws_pings_total{status="failure"})', instant=True)],
                    _grid(18, y, 6, 4), "orange",
                    thresholds=[{"color": "green", "value": None}, {"color": "orange", "value": 1}]))
    y += 4

    r["panels"] = rp
    panels.append(r)

    # ================================================================
    # Row 5: LLM Streaming Backpressure
    # ================================================================
    r = row("LLM Streaming Backpressure", y)
    y += 1
    rp = []

    rp.append(ts("LLM Stream Requests (per minute)",
                  [_prom("sum(rate(llm_stream_requests_total[1m])) by (provider) * 60", "{{provider}}")],
                  _grid(0, y, 8, 8), unit="reqps"))

    rp.append(ts("LLM Stream Chunks (per minute)",
                  [_prom("sum(rate(llm_stream_chunks_yielded_total[1m]))*60", "Yielded"),
                   _prom("sum(rate(llm_stream_chunks_dropped_total[1m]))*60", "Dropped")],
                  _grid(8, y, 8, 8), stack=True))

    rp.append(ts("LLM Stream Duration by Provider",
                  [_prom("rate(llm_stream_duration_seconds_sum[5m]) / rate(llm_stream_duration_seconds_count[5m])",
                         "Avg Duration")],
                  _grid(16, y, 8, 8), unit="s"))
    y += 8

    rp.append(stat("LLM Stream Requests",
                    [_prom("sum(llm_stream_requests_total)", instant=True)],
                    _grid(0, y, 6, 4), "blue"))
    rp.append(stat("LLM Stream Errors",
                    [_prom("sum(llm_stream_errors_total)", instant=True)],
                    _grid(6, y, 4, 4), "red",
                    thresholds=[{"color": "green", "value": None}, {"color": "red", "value": 1}]))
    rp.append(stat("Chunks Yielded",
                    [_prom("sum(llm_stream_chunks_yielded_total)", instant=True)],
                    _grid(10, y, 4, 4), "green"))
    rp.append(stat("Chunks Dropped",
                    [_prom("sum(llm_stream_chunks_dropped_total)", instant=True)],
                    _grid(14, y, 5, 4), "orange",
                    thresholds=[{"color": "green", "value": None}, {"color": "orange", "value": 1}]))
    rp.append(pie("Streams by Provider",
                   [_prom("sum(llm_stream_requests_total) by (provider)", "{{provider}}")],
                   _grid(19, y, 5, 4)))
    y += 4

    r["panels"] = rp
    panels.append(r)

    # ================================================================
    # Row 6: Streaming Logs
    # ================================================================
    r = row("Streaming Logs", y)
    y += 1
    rp = []

    rp.append(logs("SSE Stream Logs",
                    [_loki('{job="ia-agent-fwk"} |= "streaming_data" |= "sse"')],
                    _grid(0, y, 12, 8)))
    rp.append(logs("WebSocket Logs",
                    [_loki('{job="ia-agent-fwk"} |= "streaming_data" |= "ws_"')],
                    _grid(12, y, 12, 8)))
    y += 8

    rp.append(logs("LLM Stream Logs",
                    [_loki('{job="ia-agent-fwk"} |= "llm_data" |= "stream"')],
                    _grid(0, y, 12, 8)))
    rp.append(logs("Client Disconnect & Error Logs",
                    [_loki('{job="ia-agent-fwk"} |~ "disconnect|cancelled|ping failed"')],
                    _grid(12, y, 12, 8)))
    y += 8

    r["panels"] = rp
    panels.append(r)

    return {
        "dashboard": {
            "uid": DASHBOARD_UID,
            "title": "IA Agent FWK - Streaming",
            "tags": ["ia-agent-fwk", "streaming", "sse", "websocket"],
            "timezone": "browser",
            "schemaVersion": 39,
            "version": 1,
            "refresh": "30s",
            "time": {"from": "now-1h", "to": "now"},
            "panels": panels,
        },
        "overwrite": True,
    }


def main() -> None:
    dashboard = build()
    out = Path("docker/grafana/dashboards/streaming.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dashboard, indent=2) + "\n")

    panel_count = 0
    row_count = 0
    for p in dashboard["dashboard"]["panels"]:
        if p["type"] == "row":
            row_count += 1
            panel_count += len(p.get("panels", []))
        else:
            panel_count += 1
    print(f"Generated {out} ({panel_count} panels, {row_count} rows)")


if __name__ == "__main__":
    main()
