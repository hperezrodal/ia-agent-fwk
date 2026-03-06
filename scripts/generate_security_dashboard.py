#!/usr/bin/env python3
"""Generate the Security Hardening Grafana dashboard JSON.

Covers:
- Rate Limiting
- Audit Logging
- Input Sanitization

Run:
    python3 scripts/generate_security_dashboard.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# --- Constants ---
PROM_DS = {"type": "prometheus", "uid": "PBFA97CFB590B2093"}
LOKI_DS = {"type": "loki", "uid": "loki-datasource"}
TEMPO_DS = {"type": "tempo", "uid": "tempo-datasource"}
DASHBOARD_UID = "ia-agent-fwk-security"

# --- Helpers ---
_next_id = 0


def _id() -> int:
    global _next_id
    _next_id += 1
    return _next_id


def _grid(col: int, row: int, w: int = 6, h: int = 8) -> dict[str, int]:
    return {"h": h, "w": w, "x": col, "y": row}


def _prom_target(expr: str, legend: str = "", instant: bool = False) -> dict[str, Any]:
    t: dict[str, Any] = {
        "datasource": PROM_DS,
        "expr": expr,
        "legendFormat": legend,
    }
    if instant:
        t["instant"] = True
        t["range"] = False
    return t


def _loki_target(expr: str) -> dict[str, Any]:
    return {
        "datasource": LOKI_DS,
        "expr": expr,
        "queryType": "range",
    }


def stat_panel(
    title: str,
    targets: list[dict],
    pos: dict[str, int],
    color: str = "green",
    unit: str = "short",
    thresholds: list[dict] | None = None,
) -> dict[str, Any]:
    if thresholds is None:
        thresholds = [{"color": color, "value": None}]
    return {
        "id": _id(),
        "type": "stat",
        "title": title,
        "datasource": PROM_DS,
        "targets": targets,
        "gridPos": pos,
        "options": {"reduceOptions": {"calcs": ["lastNotNull"]}, "colorMode": "background"},
        "fieldConfig": {
            "defaults": {
                "unit": unit,
                "thresholds": {"mode": "absolute", "steps": thresholds},
            }
        },
    }


def ts_panel(
    title: str,
    targets: list[dict],
    pos: dict[str, int],
    unit: str = "short",
    stack: bool = False,
) -> dict[str, Any]:
    custom: dict[str, Any] = {"drawStyle": "line", "fillOpacity": 10, "lineWidth": 2}
    if stack:
        custom["stacking"] = {"mode": "normal"}
        custom["fillOpacity"] = 30
    return {
        "id": _id(),
        "type": "timeseries",
        "title": title,
        "datasource": PROM_DS,
        "targets": targets,
        "gridPos": pos,
        "fieldConfig": {
            "defaults": {"unit": unit, "custom": custom}
        },
    }


def gauge_panel(
    title: str,
    targets: list[dict],
    pos: dict[str, int],
    unit: str = "percentunit",
    thresholds: list[dict] | None = None,
) -> dict[str, Any]:
    if thresholds is None:
        thresholds = [
            {"color": "green", "value": None},
            {"color": "yellow", "value": 0.7},
            {"color": "red", "value": 0.9},
        ]
    return {
        "id": _id(),
        "type": "gauge",
        "title": title,
        "datasource": PROM_DS,
        "targets": targets,
        "gridPos": pos,
        "options": {"reduceOptions": {"calcs": ["lastNotNull"]}},
        "fieldConfig": {
            "defaults": {
                "unit": unit,
                "min": 0,
                "max": 1,
                "thresholds": {"mode": "absolute", "steps": thresholds},
            }
        },
    }


def piechart_panel(
    title: str,
    targets: list[dict],
    pos: dict[str, int],
) -> dict[str, Any]:
    return {
        "id": _id(),
        "type": "piechart",
        "title": title,
        "datasource": PROM_DS,
        "targets": targets,
        "gridPos": pos,
        "options": {
            "reduceOptions": {"calcs": ["lastNotNull"]},
            "legend": {"displayMode": "table", "placement": "right"},
            "pieType": "donut",
        },
    }


def bargauge_panel(
    title: str,
    targets: list[dict],
    pos: dict[str, int],
    unit: str = "short",
) -> dict[str, Any]:
    return {
        "id": _id(),
        "type": "bargauge",
        "title": title,
        "datasource": PROM_DS,
        "targets": targets,
        "gridPos": pos,
        "options": {
            "reduceOptions": {"calcs": ["lastNotNull"]},
            "displayMode": "gradient",
            "orientation": "horizontal",
        },
        "fieldConfig": {"defaults": {"unit": unit}},
    }


def logs_panel(
    title: str,
    targets: list[dict],
    pos: dict[str, int],
) -> dict[str, Any]:
    return {
        "id": _id(),
        "type": "logs",
        "title": title,
        "datasource": LOKI_DS,
        "targets": targets,
        "gridPos": pos,
        "options": {
            "showTime": True,
            "sortOrder": "Descending",
            "enableLogDetails": True,
            "wrapLogMessage": True,
        },
    }


def row_panel(title: str, y: int, collapsed: bool = True) -> dict[str, Any]:
    return {
        "id": _id(),
        "type": "row",
        "title": title,
        "collapsed": collapsed,
        "gridPos": {"h": 1, "w": 24, "x": 0, "y": y},
        "panels": [],
    }


# --- Build dashboard ---
def build() -> dict[str, Any]:
    panels: list[dict[str, Any]] = []
    y = 0

    # ================================================================
    # Row 1: Security Overview
    # ================================================================
    row = row_panel("Security Overview", y)
    y += 1
    rp: list[dict[str, Any]] = []

    # Stat panels
    rp.append(stat_panel(
        "Auth Success",
        [_prom_target('sum(api_auth_total{result="success"})', instant=True)],
        _grid(0, y, 4, 4),
        color="green",
    ))
    rp.append(stat_panel(
        "Auth Failures",
        [_prom_target('sum(api_auth_total{result="failure"})', instant=True)],
        _grid(4, y, 4, 4),
        color="red",
        thresholds=[
            {"color": "green", "value": None},
            {"color": "yellow", "value": 1},
            {"color": "red", "value": 10},
        ],
    ))
    rp.append(stat_panel(
        "Rate Limit Hits",
        [_prom_target("sum(api_rate_limit_exceeded_total)", instant=True)],
        _grid(8, y, 4, 4),
        color="orange",
        thresholds=[
            {"color": "green", "value": None},
            {"color": "orange", "value": 1},
            {"color": "red", "value": 50},
        ],
    ))
    rp.append(stat_panel(
        "Audit Events",
        [_prom_target("sum(audit_events_total)", instant=True)],
        _grid(12, y, 4, 4),
        color="blue",
    ))
    rp.append(stat_panel(
        "Sanitization Ops",
        [_prom_target("sum(sanitization_operations_total)", instant=True)],
        _grid(16, y, 4, 4),
        color="purple",
    ))
    rp.append(stat_panel(
        "Sensitive Detections",
        [_prom_target("sum(sanitization_detections_total)", instant=True)],
        _grid(20, y, 4, 4),
        color="orange",
        thresholds=[
            {"color": "green", "value": None},
            {"color": "yellow", "value": 1},
            {"color": "red", "value": 10},
        ],
    ))
    y += 4

    # Auth success rate gauge
    rp.append(gauge_panel(
        "Authentication Success Rate",
        [_prom_target(
            'sum(api_auth_total{result="success"}) / sum(api_auth_total)',
            instant=True,
        )],
        _grid(0, y, 6, 6),
    ))

    # Auth rate over time
    rp.append(ts_panel(
        "Authentication Rate (per minute)",
        [
            _prom_target('sum(rate(api_auth_total{result="success"}[1m]))*60', "Success"),
            _prom_target('sum(rate(api_auth_total{result="failure"}[1m]))*60', "Failure"),
        ],
        _grid(6, y, 10, 6),
        unit="reqps",
        stack=True,
    ))

    # Security events pie
    rp.append(piechart_panel(
        "Security Events by Type",
        [_prom_target("sum(audit_events_total) by (event_type)", "{{event_type}}")],
        _grid(16, y, 8, 6),
    ))
    y += 6

    row["panels"] = rp
    panels.append(row)

    # ================================================================
    # Row 2: Rate Limiting
    # ================================================================
    row = row_panel("Rate Limiting", y)
    y += 1
    rp = []

    # Rate limit checks allowed vs denied
    rp.append(ts_panel(
        "Rate Limit Checks (Allowed vs Denied)",
        [
            _prom_target('sum(rate(rate_limit_checks_total{status="allowed"}[1m]))*60', "Allowed"),
            _prom_target('sum(rate(rate_limit_checks_total{status="denied"}[1m]))*60', "Denied"),
        ],
        _grid(0, y, 8, 8),
        unit="reqps",
        stack=True,
    ))

    # Rate limit exceeded total over time
    rp.append(ts_panel(
        "Rate Limit Exceeded Over Time",
        [_prom_target("sum(rate(api_rate_limit_exceeded_total[1m]))*60", "Exceeded")],
        _grid(8, y, 8, 8),
        unit="reqps",
    ))

    # Window usage ratio
    rp.append(gauge_panel(
        "Window Usage Ratio (avg)",
        [_prom_target(
            "rate(rate_limit_window_usage_ratio_sum[5m]) / rate(rate_limit_window_usage_ratio_count[5m])",
            instant=True,
        )],
        _grid(16, y, 8, 8),
    ))
    y += 8

    # Stat row for rate limiting
    rp.append(stat_panel(
        "Total Checks Allowed",
        [_prom_target('sum(rate_limit_checks_total{status="allowed"})', instant=True)],
        _grid(0, y, 6, 4),
        color="green",
    ))
    rp.append(stat_panel(
        "Total Checks Denied",
        [_prom_target('sum(rate_limit_checks_total{status="denied"})', instant=True)],
        _grid(6, y, 6, 4),
        color="red",
        thresholds=[
            {"color": "green", "value": None},
            {"color": "orange", "value": 1},
            {"color": "red", "value": 10},
        ],
    ))
    rp.append(stat_panel(
        "API Rate Limit Allowed",
        [_prom_target("sum(api_rate_limit_allowed_total)", instant=True)],
        _grid(12, y, 6, 4),
        color="green",
    ))
    rp.append(stat_panel(
        "API Rate Limit Exceeded",
        [_prom_target("sum(api_rate_limit_exceeded_total)", instant=True)],
        _grid(18, y, 6, 4),
        color="red",
        thresholds=[
            {"color": "green", "value": None},
            {"color": "orange", "value": 1},
            {"color": "red", "value": 50},
        ],
    ))
    y += 4

    # Auth failure by reason
    rp.append(bargauge_panel(
        "Auth Failures by Reason",
        [_prom_target('sum(api_auth_total{result="failure"}) by (reason)', "{{reason}}")],
        _grid(0, y, 12, 6),
    ))

    # Denial rate over time
    rp.append(ts_panel(
        "Rate Limit Denial Rate (%)",
        [_prom_target(
            'sum(rate(rate_limit_checks_total{status="denied"}[5m])) / sum(rate(rate_limit_checks_total[5m])) * 100',
            "Denial %",
        )],
        _grid(12, y, 12, 6),
        unit="percent",
    ))
    y += 6

    row["panels"] = rp
    panels.append(row)

    # ================================================================
    # Row 3: Audit Logging
    # ================================================================
    row = row_panel("Audit Logging", y)
    y += 1
    rp = []

    # Audit events by type
    rp.append(ts_panel(
        "Audit Events Rate by Type (per minute)",
        [
            _prom_target('sum(rate(audit_events_total{event_type="auth_success"}[1m]))*60', "Auth Success"),
            _prom_target('sum(rate(audit_events_total{event_type="auth_failure"}[1m]))*60', "Auth Failure"),
            _prom_target('sum(rate(audit_events_total{event_type="agent_execution"}[1m]))*60', "Agent Exec"),
            _prom_target('sum(rate(audit_events_total{event_type="tool_execution"}[1m]))*60', "Tool Exec"),
            _prom_target('sum(rate(audit_events_total{event_type="rate_limit_hit"}[1m]))*60', "Rate Limit"),
            _prom_target('sum(rate(audit_events_total{event_type="config_change"}[1m]))*60', "Config Change"),
        ],
        _grid(0, y, 12, 8),
        unit="short",
        stack=True,
    ))

    # Audit events pie chart
    rp.append(piechart_panel(
        "Audit Events Distribution",
        [_prom_target("sum(audit_events_total) by (event_type)", "{{event_type}}")],
        _grid(12, y, 6, 8),
    ))

    # Audit events by result
    rp.append(piechart_panel(
        "Audit Events by Result",
        [_prom_target("sum(audit_events_total) by (result)", "{{result}}")],
        _grid(18, y, 6, 8),
    ))
    y += 8

    # Stats per event type
    event_types = [
        ("auth_success", "Auth Success Events", "green"),
        ("auth_failure", "Auth Failure Events", "red"),
        ("agent_execution", "Agent Execution Events", "blue"),
        ("tool_execution", "Tool Execution Events", "purple"),
        ("rate_limit_hit", "Rate Limit Hit Events", "orange"),
        ("config_change", "Config Change Events", "yellow"),
    ]
    for i, (etype, label, color) in enumerate(event_types):
        rp.append(stat_panel(
            label,
            [_prom_target(f'sum(audit_events_total{{event_type="{etype}"}})', instant=True)],
            _grid((i % 6) * 4, y + (i // 6) * 4, 4, 4),
            color=color,
        ))
    y += 4

    row["panels"] = rp
    panels.append(row)

    # ================================================================
    # Row 4: Input Sanitization
    # ================================================================
    row = row_panel("Input Sanitization", y)
    y += 1
    rp = []

    # Operations by type
    rp.append(ts_panel(
        "Sanitization Operations Rate (per minute)",
        [
            _prom_target('sum(rate(sanitization_operations_total{operation="log_value"}[1m]))*60', "Log Value"),
            _prom_target('sum(rate(sanitization_operations_total{operation="mask_secret"}[1m]))*60', "Mask Secret"),
            _prom_target('sum(rate(sanitization_operations_total{operation="error_message"}[1m]))*60', "Error Message"),
        ],
        _grid(0, y, 8, 8),
        unit="short",
        stack=True,
    ))

    # Detections by type
    rp.append(ts_panel(
        "Sanitization Detections Rate (per minute)",
        [
            _prom_target('sum(rate(sanitization_detections_total{type="control_chars"}[1m]))*60', "Control Chars"),
            _prom_target('sum(rate(sanitization_detections_total{type="truncation"}[1m]))*60', "Truncation"),
            _prom_target('sum(rate(sanitization_detections_total{type="sensitive_error"}[1m]))*60', "Sensitive Error"),
        ],
        _grid(8, y, 8, 8),
        unit="short",
        stack=True,
    ))

    # Pie chart of operations
    rp.append(piechart_panel(
        "Sanitization Operations Distribution",
        [_prom_target("sum(sanitization_operations_total) by (operation)", "{{operation}}")],
        _grid(16, y, 8, 8),
    ))
    y += 8

    # Stats
    rp.append(stat_panel(
        "Log Value Ops",
        [_prom_target('sum(sanitization_operations_total{operation="log_value"})', instant=True)],
        _grid(0, y, 4, 4),
        color="blue",
    ))
    rp.append(stat_panel(
        "Mask Secret Ops",
        [_prom_target('sum(sanitization_operations_total{operation="mask_secret"})', instant=True)],
        _grid(4, y, 4, 4),
        color="purple",
    ))
    rp.append(stat_panel(
        "Error Message Ops",
        [_prom_target('sum(sanitization_operations_total{operation="error_message"})', instant=True)],
        _grid(8, y, 4, 4),
        color="orange",
    ))
    rp.append(stat_panel(
        "Control Chars Detected",
        [_prom_target('sum(sanitization_detections_total{type="control_chars"})', instant=True)],
        _grid(12, y, 4, 4),
        color="yellow",
        thresholds=[
            {"color": "green", "value": None},
            {"color": "yellow", "value": 1},
            {"color": "red", "value": 10},
        ],
    ))
    rp.append(stat_panel(
        "Truncations",
        [_prom_target('sum(sanitization_detections_total{type="truncation"})', instant=True)],
        _grid(16, y, 4, 4),
        color="orange",
    ))
    rp.append(stat_panel(
        "Sensitive Errors Sanitized",
        [_prom_target('sum(sanitization_detections_total{type="sensitive_error"})', instant=True)],
        _grid(20, y, 4, 4),
        color="red",
        thresholds=[
            {"color": "green", "value": None},
            {"color": "yellow", "value": 1},
            {"color": "red", "value": 5},
        ],
    ))
    y += 4

    row["panels"] = rp
    panels.append(row)

    # ================================================================
    # Row 5: Security Logs
    # ================================================================
    row = row_panel("Security Logs", y)
    y += 1
    rp = []

    rp.append(logs_panel(
        "Audit Event Logs",
        [_loki_target('{job="ia-agent-fwk"} |= "audit_event"')],
        _grid(0, y, 24, 8),
    ))
    y += 8

    rp.append(logs_panel(
        "Rate Limit Logs",
        [_loki_target('{job="ia-agent-fwk"} |= "rate_limit"')],
        _grid(0, y, 12, 8),
    ))
    rp.append(logs_panel(
        "Sanitization / Security Logs",
        [_loki_target('{job="ia-agent-fwk"} |= "security_data"')],
        _grid(12, y, 12, 8),
    ))
    y += 8

    row["panels"] = rp
    panels.append(row)

    # --- Assemble dashboard ---
    return {
        "dashboard": {
            "uid": DASHBOARD_UID,
            "title": "IA Agent FWK - Security Hardening",
            "tags": ["ia-agent-fwk", "security", "rate-limiting", "audit", "sanitization"],
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
    out = Path("docker/grafana/dashboards/security-hardening.json")
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
