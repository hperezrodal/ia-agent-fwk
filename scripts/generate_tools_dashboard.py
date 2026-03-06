#!/usr/bin/env python3
"""Generate the Tools System Grafana dashboard JSON.

Covers:
- Tool Registry (registrations, lookups, size)
- Tool Execution (duration, success/error, by tool name)
- Timeouts and Error Handling
- Permission Checks
- Audit (structured logs)

Run:
    python3 scripts/generate_tools_dashboard.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROM_DS = {"type": "prometheus", "uid": "PBFA97CFB590B2093"}
LOKI_DS = {"type": "loki", "uid": "loki-datasource"}
DASHBOARD_UID = "ia-agent-fwk-tools"

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


def table(title: str, targets: list, pos: dict) -> dict[str, Any]:
    return {
        "id": _id(), "type": "table", "title": title, "datasource": PROM_DS,
        "targets": targets, "gridPos": pos,
        "options": {"sortBy": [{"displayName": "Value", "desc": True}]},
        "fieldConfig": {"defaults": {"custom": {"align": "auto"}}},
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


def build() -> dict[str, Any]:
    panels: list[dict[str, Any]] = []
    y = 0

    # ================================================================
    # Row 1: Tools Overview
    # ================================================================
    r = row("Tools Overview", y)
    y += 1
    rp: list[dict[str, Any]] = []

    rp.append(stat("Total Executions",
                    [_prom("sum(tool_executions_total)", instant=True)],
                    _grid(0, y, 4, 4), "blue"))
    rp.append(stat("Successful",
                    [_prom('sum(tool_executions_total{status="success"})', instant=True)],
                    _grid(4, y, 4, 4), "green"))
    rp.append(stat("Errors",
                    [_prom('sum(tool_executions_total{status="error"})', instant=True)],
                    _grid(8, y, 4, 4), "red",
                    thresholds=[{"color": "green", "value": None}, {"color": "orange", "value": 1}, {"color": "red", "value": 10}]))
    rp.append(stat("Registry Size",
                    [_prom("max(tool_registry_size_sum) / max(tool_registry_size_count)", instant=True)],
                    _grid(12, y, 4, 4), "purple"))
    rp.append(stat("Permission Denied",
                    [_prom('sum(tool_permission_checks_total{result="denied"})', instant=True)],
                    _grid(16, y, 4, 4), "orange",
                    thresholds=[{"color": "green", "value": None}, {"color": "orange", "value": 1}]))
    rp.append(stat("Avg Duration",
                    [_prom("rate(tool_execution_duration_seconds_sum[5m]) / rate(tool_execution_duration_seconds_count[5m])",
                           instant=True)],
                    _grid(20, y, 4, 4), "blue", unit="s"))
    y += 4

    # Rate & distribution
    rp.append(ts("Tool Executions Rate (per minute)",
                  [_prom('sum(rate(tool_executions_total{status="success"}[1m]))*60', "Success"),
                   _prom('sum(rate(tool_executions_total{status="error"}[1m]))*60', "Error")],
                  _grid(0, y, 8, 8), unit="reqps", stack=True))

    rp.append(pie("Executions by Tool",
                   [_prom("sum(tool_executions_total) by (tool)", "{{tool}}")],
                   _grid(8, y, 8, 8)))

    rp.append(pie("Errors by Type",
                   [_prom("sum(tool_errors_total) by (error_type)", "{{error_type}}")],
                   _grid(16, y, 8, 8)))
    y += 8

    r["panels"] = rp
    panels.append(r)

    # ================================================================
    # Row 2: Tool Registry
    # ================================================================
    r = row("Tool Registry", y)
    y += 1
    rp = []

    rp.append(stat("Total Registrations",
                    [_prom("sum(tool_registry_registrations_total)", instant=True)],
                    _grid(0, y, 6, 4), "blue"))
    rp.append(stat("Registry Lookups (hits)",
                    [_prom('sum(tool_registry_lookups_total{status="hit"})', instant=True)],
                    _grid(6, y, 6, 4), "green"))
    rp.append(stat("Registry Lookups (misses)",
                    [_prom('sum(tool_registry_lookups_total{status="miss"})', instant=True)],
                    _grid(12, y, 6, 4), "orange",
                    thresholds=[{"color": "green", "value": None}, {"color": "orange", "value": 1}]))
    rp.append(stat("Tool Removals",
                    [_prom("sum(tool_registry_removals_total)", instant=True)],
                    _grid(18, y, 6, 4), "yellow"))
    y += 4

    rp.append(ts("Registry Lookups Rate (per minute)",
                  [_prom('sum(rate(tool_registry_lookups_total{status="hit"}[1m]))*60', "Hits"),
                   _prom('sum(rate(tool_registry_lookups_total{status="miss"}[1m]))*60', "Misses")],
                  _grid(0, y, 12, 8), stack=True))

    rp.append(ts("Registry Size Over Time",
                  [_prom("rate(tool_registry_size_sum[5m]) / rate(tool_registry_size_count[5m])",
                         "Registry Size")],
                  _grid(12, y, 12, 8)))
    y += 8

    r["panels"] = rp
    panels.append(r)

    # ================================================================
    # Row 3: Tool Execution Performance
    # ================================================================
    r = row("Tool Execution with Timeout & Error Handling", y)
    y += 1
    rp = []

    rp.append(ts("Execution Duration by Tool (avg, 5m)",
                  [_prom("rate(tool_execution_duration_seconds_sum[5m]) / rate(tool_execution_duration_seconds_count[5m])",
                         "Avg (all tools)"),
                   _prom('topk(5, rate(tool_execution_duration_seconds_sum[5m]) / rate(tool_execution_duration_seconds_count[5m]))',
                         "{{tool}}")],
                  _grid(0, y, 12, 8), unit="s"))

    rp.append(ts("Execution Rate by Tool (per minute)",
                  [_prom("sum(rate(tool_executions_total[1m])) by (tool) * 60", "{{tool}}")],
                  _grid(12, y, 12, 8), unit="reqps"))
    y += 8

    # Error breakdown
    rp.append(ts("Errors by Type Over Time (per minute)",
                  [_prom('sum(rate(tool_errors_total{error_type="timeout"}[1m]))*60', "Timeout"),
                   _prom('sum(rate(tool_errors_total{error_type="permission"}[1m]))*60', "Permission"),
                   _prom('sum(rate(tool_errors_total{error_type="validation"}[1m]))*60', "Validation"),
                   _prom('sum(rate(tool_errors_total{error_type="notfound"}[1m]))*60', "Not Found"),
                   _prom('sum(rate(tool_errors_total{error_type="execution"}[1m]))*60', "Execution"),
                   _prom('sum(rate(tool_errors_total{error_type="unknown"}[1m]))*60', "Unknown")],
                  _grid(0, y, 12, 8), stack=True))

    rp.append(bar("Errors by Tool (top 10)",
                   [_prom("topk(10, sum(tool_errors_total) by (tool))", "{{tool}}")],
                   _grid(12, y, 12, 8)))
    y += 8

    # Success rate and stats
    rp.append(stat("Success Rate",
                    [_prom('sum(tool_executions_total{status="success"}) / sum(tool_executions_total)', instant=True)],
                    _grid(0, y, 6, 4), "green", unit="percentunit",
                    thresholds=[{"color": "red", "value": None}, {"color": "yellow", "value": 0.9}, {"color": "green", "value": 0.95}]))
    rp.append(stat("Timeout Errors",
                    [_prom('sum(tool_errors_total{error_type="timeout"})', instant=True)],
                    _grid(6, y, 6, 4), "orange",
                    thresholds=[{"color": "green", "value": None}, {"color": "orange", "value": 1}, {"color": "red", "value": 5}]))
    rp.append(stat("Validation Errors",
                    [_prom('sum(tool_errors_total{error_type="validation"})', instant=True)],
                    _grid(12, y, 6, 4), "yellow",
                    thresholds=[{"color": "green", "value": None}, {"color": "yellow", "value": 1}]))
    rp.append(stat("Execution Errors",
                    [_prom('sum(tool_errors_total{error_type="execution"})', instant=True)],
                    _grid(18, y, 6, 4), "red",
                    thresholds=[{"color": "green", "value": None}, {"color": "red", "value": 1}]))
    y += 4

    r["panels"] = rp
    panels.append(r)

    # ================================================================
    # Row 4: Permission System & Audit
    # ================================================================
    r = row("Permission System & Audit", y)
    y += 1
    rp = []

    rp.append(ts("Permission Checks Rate (per minute)",
                  [_prom('sum(rate(tool_permission_checks_total{result="allowed"}[1m]))*60', "Allowed"),
                   _prom('sum(rate(tool_permission_checks_total{result="denied"}[1m]))*60', "Denied")],
                  _grid(0, y, 8, 8), stack=True))

    rp.append(pie("Permission Checks by Mode",
                   [_prom("sum(tool_permission_checks_total) by (mode)", "{{mode}}")],
                   _grid(8, y, 8, 8)))

    rp.append(pie("Permission Results",
                   [_prom("sum(tool_permission_checks_total) by (result)", "{{result}}")],
                   _grid(16, y, 8, 8)))
    y += 8

    rp.append(stat("Allowed Checks",
                    [_prom('sum(tool_permission_checks_total{result="allowed"})', instant=True)],
                    _grid(0, y, 6, 4), "green"))
    rp.append(stat("Denied Checks",
                    [_prom('sum(tool_permission_checks_total{result="denied"})', instant=True)],
                    _grid(6, y, 6, 4), "red",
                    thresholds=[{"color": "green", "value": None}, {"color": "orange", "value": 1}]))
    rp.append(stat("Not Found Errors",
                    [_prom('sum(tool_errors_total{error_type="notfound"})', instant=True)],
                    _grid(12, y, 6, 4), "orange",
                    thresholds=[{"color": "green", "value": None}, {"color": "orange", "value": 1}]))
    rp.append(stat("Permission Errors",
                    [_prom('sum(tool_errors_total{error_type="permission"})', instant=True)],
                    _grid(18, y, 6, 4), "red",
                    thresholds=[{"color": "green", "value": None}, {"color": "red", "value": 1}]))
    y += 4

    r["panels"] = rp
    panels.append(r)

    # ================================================================
    # Row 5: Tool Execution Logs
    # ================================================================
    r = row("Tool Execution Logs", y)
    y += 1
    rp = []

    rp.append(logs("Tool Execution Logs (all)",
                    [_loki('{job="ia-agent-fwk"} |= "tool_data"')],
                    _grid(0, y, 24, 8)))
    y += 8

    rp.append(logs("Tool Error Logs",
                    [_loki('{job="ia-agent-fwk"} |= "tool_data" |= "error"')],
                    _grid(0, y, 12, 8)))
    rp.append(logs("Tool Permission / Timeout Logs",
                    [_loki('{job="ia-agent-fwk"} |= "tool_data" |~ "permission|timeout"')],
                    _grid(12, y, 12, 8)))
    y += 8

    r["panels"] = rp
    panels.append(r)

    return {
        "dashboard": {
            "uid": DASHBOARD_UID,
            "title": "IA Agent FWK - Tools System",
            "tags": ["ia-agent-fwk", "tools", "registry", "execution", "permissions"],
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
    out = Path("docker/grafana/dashboards/tools-system.json")
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
