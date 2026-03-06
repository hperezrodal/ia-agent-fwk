#!/usr/bin/env python3
"""Generate Grafana dashboard JSON for Multi-Agent Orchestration observability."""

import json

PROM_DS = {"type": "prometheus", "uid": "PBFA97CFB590B2093"}
LOKI_DS = {"type": "loki", "uid": "loki-datasource"}

_panel_id = 0


def _next_id():
    global _panel_id
    _panel_id += 1
    return _panel_id


def row(title, y, collapsed=False):
    return {
        "id": _next_id(),
        "type": "row",
        "title": title,
        "gridPos": {"h": 1, "w": 24, "x": 0, "y": y},
        "collapsed": collapsed,
        "panels": [],
    }


def stat_panel(title, expr, x, y, w=6, h=4, unit="short", color="green"):
    return {
        "id": _next_id(),
        "type": "stat",
        "title": title,
        "datasource": PROM_DS,
        "gridPos": {"h": h, "w": w, "x": x, "y": y},
        "targets": [{"expr": expr, "refId": "A"}],
        "fieldConfig": {
            "defaults": {
                "unit": unit,
                "thresholds": {
                    "mode": "absolute",
                    "steps": [{"color": color, "value": None}],
                },
            }
        },
        "options": {"colorMode": "background", "graphMode": "area", "textMode": "auto"},
    }


def timeseries(title, targets, x, y, w=12, h=8, unit="short", legend_mode="list"):
    t = []
    for i, (expr, legend) in enumerate(targets):
        t.append({
            "expr": expr,
            "legendFormat": legend,
            "refId": chr(65 + i),
        })
    return {
        "id": _next_id(),
        "type": "timeseries",
        "title": title,
        "datasource": PROM_DS,
        "gridPos": {"h": h, "w": w, "x": x, "y": y},
        "targets": t,
        "fieldConfig": {
            "defaults": {
                "unit": unit,
                "custom": {
                    "drawStyle": "line",
                    "lineWidth": 2,
                    "fillOpacity": 15,
                    "pointSize": 5,
                    "showPoints": "auto",
                },
            }
        },
        "options": {"legend": {"displayMode": legend_mode, "placement": "bottom"}},
    }


def bargauge(title, expr, x, y, w=6, h=8, unit="short", orientation="horizontal"):
    return {
        "id": _next_id(),
        "type": "bargauge",
        "title": title,
        "datasource": PROM_DS,
        "gridPos": {"h": h, "w": w, "x": x, "y": y},
        "targets": [{"expr": expr, "legendFormat": "{{route_key}}", "refId": "A"}],
        "fieldConfig": {
            "defaults": {
                "unit": unit,
                "thresholds": {
                    "mode": "absolute",
                    "steps": [
                        {"color": "green", "value": None},
                        {"color": "yellow", "value": 5},
                        {"color": "red", "value": 10},
                    ],
                },
            }
        },
        "options": {"orientation": orientation, "displayMode": "gradient"},
    }


def piechart(title, targets, x, y, w=6, h=8):
    t = []
    for i, (expr, legend) in enumerate(targets):
        t.append({"expr": expr, "legendFormat": legend, "refId": chr(65 + i)})
    return {
        "id": _next_id(),
        "type": "piechart",
        "title": title,
        "datasource": PROM_DS,
        "gridPos": {"h": h, "w": w, "x": x, "y": y},
        "targets": t,
        "options": {
            "legend": {"displayMode": "table", "placement": "right"},
            "pieType": "donut",
        },
    }


def table_panel(title, targets, x, y, w=12, h=8):
    t = []
    for i, (expr, legend) in enumerate(targets):
        t.append({"expr": expr, "legendFormat": legend, "refId": chr(65 + i), "instant": True, "format": "table"})
    return {
        "id": _next_id(),
        "type": "table",
        "title": title,
        "datasource": PROM_DS,
        "gridPos": {"h": h, "w": w, "x": x, "y": y},
        "targets": t,
        "options": {"showHeader": True},
    }


def logs_panel(title, expr, x, y, w=24, h=8):
    return {
        "id": _next_id(),
        "type": "logs",
        "title": title,
        "datasource": LOKI_DS,
        "gridPos": {"h": h, "w": w, "x": x, "y": y},
        "targets": [{"expr": expr, "refId": "A"}],
        "options": {"showTime": True, "wrapLogMessage": True, "enableLogDetails": True, "sortOrder": "Descending"},
    }


def build_dashboard():
    panels = []
    y = 0

    # ===== ROW 1: Workflow Overview =====
    panels.append(row("Workflow Overview", y))
    y += 1

    # Total workflow executions by type
    panels.append(stat_panel(
        "Total Workflow Executions",
        'sum(workflow_executions_total)',
        0, y, w=4, color="blue",
    ))
    panels.append(stat_panel(
        "Sequential Executions",
        'sum(workflow_executions_total{type="sequential"})',
        4, y, w=4, color="purple",
    ))
    panels.append(stat_panel(
        "Parallel Executions",
        'sum(workflow_executions_total{type="parallel"})',
        8, y, w=4, color="orange",
    ))
    panels.append(stat_panel(
        "Conditional Executions",
        'sum(workflow_executions_total{type="conditional"})',
        12, y, w=4, color="blue",
    ))
    panels.append(stat_panel(
        "Workflow Builds",
        'sum(workflow_builds_total)',
        16, y, w=4, color="green",
    ))
    panels.append(stat_panel(
        "Build Errors",
        'sum(workflow_build_errors_total)',
        20, y, w=4, color="red",
    ))
    y += 4

    # Workflow executions over time by type
    panels.append(timeseries(
        "Workflow Executions/min by Type",
        [
            ('sum(rate(workflow_executions_total{type="sequential"}[5m])) * 60', "Sequential"),
            ('sum(rate(workflow_executions_total{type="parallel"}[5m])) * 60', "Parallel"),
            ('sum(rate(workflow_executions_total{type="conditional"}[5m])) * 60', "Conditional"),
        ],
        0, y, w=12, h=8, unit="ops",
    ))

    # Workflow success vs failure rate
    panels.append(timeseries(
        "Workflow Success vs Failure Rate",
        [
            ('sum(rate(workflow_executions_total{status="success"}[5m])) * 60', "Success"),
            ('sum(rate(workflow_executions_total{status="failure"}[5m])) * 60', "Failure"),
            ('sum(rate(workflow_executions_total{status="timeout"}[5m])) * 60', "Timeout"),
        ],
        12, y, w=12, h=8, unit="ops",
    ))
    y += 8

    # ===== ROW 2: Sequential Workflow =====
    panels.append(row("Sequential Workflow (Chain)", y))
    y += 1

    # Sequential duration
    panels.append(timeseries(
        "Sequential Duration (p50 / p99)",
        [
            ('histogram_quantile(0.5, sum(rate(workflow_duration_seconds_bucket{type="sequential"}[5m])) by (le))', "p50"),
            ('histogram_quantile(0.99, sum(rate(workflow_duration_seconds_bucket{type="sequential"}[5m])) by (le))', "p99"),
        ],
        0, y, w=8, h=8, unit="s",
    ))

    # Steps completed per workflow
    panels.append(timeseries(
        "Steps Completed per Workflow",
        [
            ('workflow_steps_completed_sum{type="sequential"} / workflow_steps_completed_count{type="sequential"}', "Avg Steps"),
        ],
        8, y, w=8, h=8,
    ))

    # Step-level duration
    panels.append(timeseries(
        "Step Duration by Name",
        [
            ('workflow_step_duration_seconds_sum{type="sequential"} / workflow_step_duration_seconds_count{type="sequential"}', "{{step_name}}"),
        ],
        16, y, w=8, h=8, unit="s",
    ))
    y += 8

    # Retry & fallback metrics
    panels.append(stat_panel(
        "Step Retries",
        'sum(workflow_step_retries_total{type="sequential"})',
        0, y, w=6, color="yellow",
    ))
    panels.append(stat_panel(
        "Step Fallbacks",
        'sum(workflow_step_fallbacks_total{type="sequential"})',
        6, y, w=6, color="orange",
    ))
    panels.append(stat_panel(
        "Step Failures",
        'sum(workflow_step_failures_total{type="sequential"})',
        12, y, w=6, color="red",
    ))
    panels.append(stat_panel(
        "Tokens (Sequential)",
        'sum(workflow_tokens_total_sum{type="sequential"})',
        18, y, w=6, color="blue", unit="short",
    ))
    y += 4

    # ===== ROW 3: Parallel Workflow =====
    panels.append(row("Parallel Workflow (Fan-Out / Fan-In)", y))
    y += 1

    # Parallel duration
    panels.append(timeseries(
        "Parallel Duration (p50 / p99)",
        [
            ('histogram_quantile(0.5, sum(rate(workflow_duration_seconds_bucket{type="parallel"}[5m])) by (le))', "p50"),
            ('histogram_quantile(0.99, sum(rate(workflow_duration_seconds_bucket{type="parallel"}[5m])) by (le))', "p99"),
        ],
        0, y, w=8, h=8, unit="s",
    ))

    # Fan-out size distribution
    panels.append(timeseries(
        "Fan-Out Size (Avg Parallel Steps)",
        [
            ('workflow_parallel_fan_out_size_sum / workflow_parallel_fan_out_size_count', "Avg Fan-Out"),
        ],
        8, y, w=8, h=8,
    ))

    # Parallel step durations
    panels.append(timeseries(
        "Per-Step Duration (Parallel)",
        [
            ('workflow_step_duration_seconds_sum{type="parallel"} / workflow_step_duration_seconds_count{type="parallel"}', "{{step_name}}"),
        ],
        16, y, w=8, h=8, unit="s",
    ))
    y += 8

    # Partial failure and policy stats
    panels.append(stat_panel(
        "Partial Failures",
        'sum(workflow_parallel_partial_failures_total)',
        0, y, w=6, color="orange",
    ))
    panels.append(stat_panel(
        "Step Failures (Parallel)",
        'sum(workflow_step_failures_total{type="parallel"})',
        6, y, w=6, color="red",
    ))
    panels.append(stat_panel(
        "Steps Completed (Parallel)",
        'sum(workflow_steps_completed_sum{type="parallel"})',
        12, y, w=6, color="green",
    ))
    panels.append(stat_panel(
        "Tokens (Parallel)",
        'sum(workflow_tokens_total_sum{type="parallel"})',
        18, y, w=6, color="blue",
    ))
    y += 4

    # ===== ROW 4: Conditional Routing =====
    panels.append(row("Conditional Routing", y))
    y += 1

    # Route distribution
    panels.append(bargauge(
        "Route Distribution",
        'sum by (route_key) (workflow_conditional_routes_total)',
        0, y, w=8, h=8,
    ))

    # Conditional duration
    panels.append(timeseries(
        "Conditional Duration (p50 / p99)",
        [
            ('histogram_quantile(0.5, sum(rate(workflow_duration_seconds_bucket{type="conditional"}[5m])) by (le))', "p50"),
            ('histogram_quantile(0.99, sum(rate(workflow_duration_seconds_bucket{type="conditional"}[5m])) by (le))', "p99"),
        ],
        8, y, w=8, h=8, unit="s",
    ))

    # Route trends over time
    panels.append(timeseries(
        "Route Selection Rate",
        [
            ('sum(rate(workflow_conditional_routes_total[5m])) by (route_key) * 60', "{{route_key}}"),
        ],
        16, y, w=8, h=8, unit="ops",
    ))
    y += 8

    panels.append(stat_panel(
        "Default Route Used",
        'sum(workflow_conditional_default_route_total)',
        0, y, w=6, color="yellow",
    ))
    panels.append(stat_panel(
        "No Route Found",
        'sum(workflow_conditional_no_route_total)',
        6, y, w=6, color="red",
    ))
    panels.append(stat_panel(
        "Tokens (Conditional)",
        'sum(workflow_tokens_total_sum{type="conditional"})',
        12, y, w=6, color="blue",
    ))
    panels.append(stat_panel(
        "Conditional Executions",
        'sum(workflow_executions_total{type="conditional"})',
        18, y, w=6, color="green",
    ))
    y += 4

    # ===== ROW 5: Supervisor Agent =====
    panels.append(row("Supervisor Agent (Dynamic Delegation)", y))
    y += 1

    # Delegations by sub-agent
    panels.append(timeseries(
        "Delegations/min by Sub-Agent",
        [
            ('sum(rate(supervisor_delegations_total[5m])) by (sub_agent) * 60', "{{sub_agent}}"),
        ],
        0, y, w=8, h=8, unit="ops",
    ))

    # Delegation pie chart
    panels.append(piechart(
        "Delegation Distribution",
        [
            ('sum by (sub_agent) (supervisor_delegations_total)', "{{sub_agent}}"),
        ],
        8, y, w=8, h=8,
    ))

    # Agent tool duration by agent
    panels.append(timeseries(
        "Delegation Duration by Agent",
        [
            ('agent_tool_duration_seconds_sum / agent_tool_duration_seconds_count', "{{agent_name}}"),
        ],
        16, y, w=8, h=8, unit="s",
    ))
    y += 8

    panels.append(stat_panel(
        "Total Delegations",
        'sum(supervisor_delegations_total)',
        0, y, w=4, color="blue",
    ))
    panels.append(stat_panel(
        "Supervisors Created",
        'sum(supervisor_instances_created_total)',
        4, y, w=4, color="green",
    ))
    panels.append(stat_panel(
        "Depth Exceeded",
        'sum(agent_tool_executions_total{status="depth_exceeded"})',
        8, y, w=4, color="red",
    ))
    panels.append(stat_panel(
        "Delegation Failures",
        'sum(agent_tool_executions_total{status="failure"})',
        12, y, w=4, color="orange",
    ))
    panels.append(stat_panel(
        "Delegation Success",
        'sum(agent_tool_executions_total{status="success"})',
        16, y, w=4, color="green",
    ))
    panels.append(stat_panel(
        "Avg Sub-Agent Count",
        'supervisor_sub_agents_count_sum / supervisor_sub_agents_count_count',
        20, y, w=4, color="purple",
    ))
    y += 4

    # ===== ROW 6: Error Handling in Multi-Agent Flows =====
    panels.append(row("Error Handling in Multi-Agent Flows", y))
    y += 1

    # Errors over time by type
    panels.append(timeseries(
        "Workflow Failures/min by Type",
        [
            ('sum(rate(workflow_executions_total{status="failure"}[5m])) by (type) * 60', "{{type}} failure"),
            ('sum(rate(workflow_executions_total{status="timeout"}[5m])) by (type) * 60', "{{type}} timeout"),
        ],
        0, y, w=12, h=8, unit="ops",
    ))

    # Step failures by type and step
    panels.append(timeseries(
        "Step Failures/min by Workflow Type",
        [
            ('sum(rate(workflow_step_failures_total[5m])) by (type) * 60', "{{type}}"),
        ],
        12, y, w=12, h=8, unit="ops",
    ))
    y += 8

    # Error breakdown stats
    panels.append(timeseries(
        "Retries & Fallbacks Over Time",
        [
            ('sum(rate(workflow_step_retries_total[5m])) * 60', "Retries/min"),
            ('sum(rate(workflow_step_fallbacks_total[5m])) * 60', "Fallbacks/min"),
        ],
        0, y, w=8, h=8, unit="ops",
    ))

    # Delegation failures over time
    panels.append(timeseries(
        "Agent Tool Failures & Depth Exceeded",
        [
            ('sum(rate(agent_tool_executions_total{status="failure"}[5m])) * 60', "Failures/min"),
            ('sum(rate(agent_tool_executions_total{status="depth_exceeded"}[5m])) * 60', "Depth Exceeded/min"),
        ],
        8, y, w=8, h=8, unit="ops",
    ))

    # Success rate gauge
    panels.append(timeseries(
        "Workflow Success Rate (%)",
        [
            ('sum(rate(workflow_executions_total{status="success"}[5m])) / sum(rate(workflow_executions_total[5m])) * 100', "Success %"),
        ],
        16, y, w=8, h=8, unit="percent",
    ))
    y += 8

    # ===== ROW 7: Workflow Definition & Tokens =====
    panels.append(row("Workflow Definition & Token Usage", y))
    y += 1

    # Builds over time
    panels.append(timeseries(
        "Workflow Builds/min by Type",
        [
            ('sum(rate(workflow_builds_total[5m])) by (type) * 60', "{{type}}"),
        ],
        0, y, w=8, h=8, unit="ops",
    ))

    # Token usage by workflow type
    panels.append(timeseries(
        "Token Usage by Workflow Type",
        [
            ('sum(rate(workflow_tokens_total_sum[5m])) by (type) * 60', "{{type}}"),
        ],
        8, y, w=8, h=8, unit="short",
    ))

    # Build errors
    panels.append(timeseries(
        "Build Errors Over Time",
        [
            ('sum(rate(workflow_build_errors_total[5m])) * 60', "Build Errors/min"),
        ],
        16, y, w=8, h=8, unit="ops",
    ))
    y += 8

    # ===== ROW 8: Orchestration Logs =====
    panels.append(row("Orchestration Logs", y))
    y += 1

    panels.append(logs_panel(
        "Orchestration Events (Loki)",
        '{service=~"api|worker"} |= "orchestration_data"',
        0, y, w=24, h=10,
    ))
    y += 10

    # Build dashboard JSON
    dashboard = {
        "uid": "ia-agent-fwk-orchestration",
        "title": "IA Agent FWK - Multi-Agent Orchestration",
        "tags": ["ia-agent-fwk", "orchestration", "multi-agent"],
        "timezone": "browser",
        "schemaVersion": 39,
        "version": 1,
        "refresh": "30s",
        "time": {"from": "now-1h", "to": "now"},
        "panels": panels,
        "templating": {"list": []},
        "annotations": {"list": []},
    }

    print(f"Generated {len(panels)} panels, max y={y}")
    for p in panels:
        if p["type"] == "row":
            print(f"  y={p['gridPos']['y']:>2} ... {p['title']}")

    return dashboard


if __name__ == "__main__":
    dashboard = build_dashboard()
    out = "docker/grafana/dashboards/multi-agent-orchestration.json"
    with open(out, "w") as f:
        json.dump(dashboard, f, indent=2)
    print(f"\nDashboard written to {out}")
