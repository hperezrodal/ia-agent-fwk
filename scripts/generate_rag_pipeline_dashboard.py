#!/usr/bin/env python3
"""Generate Grafana dashboard JSON for RAG Pipeline observability."""

import json

PROM_DS = {"type": "prometheus", "uid": "PBFA97CFB590B2093"}
LOKI_DS = {"type": "loki", "uid": "loki-datasource"}

_panel_id = 0


def _next_id():
    global _panel_id
    _panel_id += 1
    return _panel_id


def row(title, y):
    return {
        "id": _next_id(), "type": "row", "title": title,
        "gridPos": {"h": 1, "w": 24, "x": 0, "y": y},
        "collapsed": False, "panels": [],
    }


def stat_panel(title, expr, x, y, w=6, h=4, unit="short", color="green"):
    return {
        "id": _next_id(), "type": "stat", "title": title,
        "datasource": PROM_DS, "gridPos": {"h": h, "w": w, "x": x, "y": y},
        "targets": [{"expr": expr, "refId": "A"}],
        "fieldConfig": {"defaults": {"unit": unit, "thresholds": {"mode": "absolute", "steps": [{"color": color, "value": None}]}}},
        "options": {"colorMode": "background", "graphMode": "area", "textMode": "auto"},
    }


def ts(title, targets, x, y, w=12, h=8, unit="short"):
    t = [{"expr": e, "legendFormat": l, "refId": chr(65 + i)} for i, (e, l) in enumerate(targets)]
    return {
        "id": _next_id(), "type": "timeseries", "title": title,
        "datasource": PROM_DS, "gridPos": {"h": h, "w": w, "x": x, "y": y},
        "targets": t,
        "fieldConfig": {"defaults": {"unit": unit, "custom": {"drawStyle": "line", "lineWidth": 2, "fillOpacity": 15, "pointSize": 5, "showPoints": "auto"}}},
        "options": {"legend": {"displayMode": "list", "placement": "bottom"}},
    }


def piechart(title, targets, x, y, w=6, h=8):
    t = [{"expr": e, "legendFormat": l, "refId": chr(65 + i)} for i, (e, l) in enumerate(targets)]
    return {
        "id": _next_id(), "type": "piechart", "title": title,
        "datasource": PROM_DS, "gridPos": {"h": h, "w": w, "x": x, "y": y},
        "targets": t,
        "options": {"legend": {"displayMode": "table", "placement": "right"}, "pieType": "donut"},
    }


def bargauge(title, expr, x, y, w=6, h=8, unit="short", legend="{{strategy}}"):
    return {
        "id": _next_id(), "type": "bargauge", "title": title,
        "datasource": PROM_DS, "gridPos": {"h": h, "w": w, "x": x, "y": y},
        "targets": [{"expr": expr, "legendFormat": legend, "refId": "A"}],
        "fieldConfig": {"defaults": {"unit": unit, "thresholds": {"mode": "absolute", "steps": [{"color": "green", "value": None}, {"color": "yellow", "value": 5}, {"color": "red", "value": 10}]}}},
        "options": {"orientation": "horizontal", "displayMode": "gradient"},
    }


def logs_panel(title, expr, x, y, w=24, h=8):
    return {
        "id": _next_id(), "type": "logs", "title": title,
        "datasource": LOKI_DS, "gridPos": {"h": h, "w": w, "x": x, "y": y},
        "targets": [{"expr": expr, "refId": "A"}],
        "options": {"showTime": True, "wrapLogMessage": True, "enableLogDetails": True, "sortOrder": "Descending"},
    }


def build_dashboard():
    panels = []
    y = 0

    # ===== ROW 1: RAG Pipeline Overview =====
    panels.append(row("RAG Pipeline Overview", y)); y += 1

    panels.append(stat_panel("Documents Ingested", 'sum(rag_pipeline_ingest_total{status="success"})', 0, y, w=4, color="blue"))
    panels.append(stat_panel("Ingest Failures", 'sum(rag_pipeline_ingest_total{status="failure"})', 4, y, w=4, color="red"))
    panels.append(stat_panel("Queries Executed", 'sum(rag_pipeline_query_total{status="success"})', 8, y, w=4, color="green"))
    panels.append(stat_panel("Query Failures", 'sum(rag_pipeline_query_total{status="failure"})', 12, y, w=4, color="red"))
    panels.append(stat_panel("Embedding Requests", 'sum(rag_embedding_requests_total)', 16, y, w=4, color="purple"))
    panels.append(stat_panel("Qdrant Operations", 'sum(qdrant_operations_total)', 20, y, w=4, color="orange"))
    y += 4

    panels.append(ts("Ingest Rate/min", [
        ('sum(rate(rag_pipeline_ingest_total{status="success"}[5m])) * 60', "Success"),
        ('sum(rate(rag_pipeline_ingest_total{status="failure"}[5m])) * 60', "Failure"),
    ], 0, y, w=12, unit="ops"))
    panels.append(ts("Query Rate/min", [
        ('sum(rate(rag_pipeline_query_total{status="success"}[5m])) * 60', "Success"),
        ('sum(rate(rag_pipeline_query_total{status="failure"}[5m])) * 60', "Failure"),
    ], 12, y, w=12, unit="ops"))
    y += 8

    # ===== ROW 2: Text Chunking =====
    panels.append(row("Text Chunking", y)); y += 1

    panels.append(ts("Chunking Duration by Strategy", [
        ('rag_chunking_duration_seconds_sum{strategy="fixed"} / rag_chunking_duration_seconds_count{strategy="fixed"}', "Fixed"),
        ('rag_chunking_duration_seconds_sum{strategy="recursive"} / rag_chunking_duration_seconds_count{strategy="recursive"}', "Recursive"),
        ('rag_chunking_duration_seconds_sum{strategy="semantic"} / rag_chunking_duration_seconds_count{strategy="semantic"}', "Semantic"),
    ], 0, y, w=8, unit="s"))

    panels.append(ts("Chunks Produced per Document", [
        ('rag_chunks_produced_sum{strategy="fixed"} / rag_chunks_produced_count{strategy="fixed"}', "Fixed avg"),
        ('rag_chunks_produced_sum{strategy="recursive"} / rag_chunks_produced_count{strategy="recursive"}', "Recursive avg"),
        ('rag_chunks_produced_sum{strategy="semantic"} / rag_chunks_produced_count{strategy="semantic"}', "Semantic avg"),
    ], 8, y, w=8))

    panels.append(ts("Avg Chunk Size (chars)", [
        ('rag_chunk_size_chars_sum{strategy="fixed"} / rag_chunk_size_chars_count{strategy="fixed"}', "Fixed"),
        ('rag_chunk_size_chars_sum{strategy="recursive"} / rag_chunk_size_chars_count{strategy="recursive"}', "Recursive"),
        ('rag_chunk_size_chars_sum{strategy="semantic"} / rag_chunk_size_chars_count{strategy="semantic"}', "Semantic"),
    ], 16, y, w=8, unit="short"))
    y += 8

    panels.append(piechart("Chunking Strategy Distribution", [
        ('sum(rag_chunking_total{strategy="fixed"})', "Fixed"),
        ('sum(rag_chunking_total{strategy="recursive"})', "Recursive"),
        ('sum(rag_chunking_total{strategy="semantic"})', "Semantic"),
    ], 0, y, w=8))
    panels.append(stat_panel("Total Chunks Produced (Fixed)", 'sum(rag_chunks_produced_sum{strategy="fixed"})', 8, y, w=4, color="blue"))
    panels.append(stat_panel("Total Chunks Produced (Recursive)", 'sum(rag_chunks_produced_sum{strategy="recursive"})', 12, y, w=4, color="purple"))
    panels.append(stat_panel("Total Chunks Produced (Semantic)", 'sum(rag_chunks_produced_sum{strategy="semantic"})', 16, y, w=4, color="orange"))
    panels.append(stat_panel("Semantic Sentences Total", 'sum(rag_semantic_sentences_total_sum)', 20, y, w=4, color="green"))
    y += 8

    # ===== ROW 3: Embedding Generation =====
    panels.append(row("Embedding Generation", y)); y += 1

    panels.append(ts("Embedding Requests/min by Provider", [
        ('sum(rate(rag_embedding_requests_total{provider="ollama"}[5m])) * 60', "Ollama"),
        ('sum(rate(rag_embedding_requests_total{provider="openai"}[5m])) * 60', "OpenAI"),
    ], 0, y, w=8, unit="ops"))

    panels.append(ts("Embedding Duration by Provider", [
        ('rag_embedding_duration_seconds_sum{provider="ollama"} / rag_embedding_duration_seconds_count{provider="ollama"}', "Ollama avg"),
        ('rag_embedding_duration_seconds_sum{provider="openai"} / rag_embedding_duration_seconds_count{provider="openai"}', "OpenAI avg"),
    ], 8, y, w=8, unit="s"))

    panels.append(ts("Batch Size (texts per request)", [
        ('rag_embedding_texts_count_sum{provider="ollama"} / rag_embedding_texts_count_count{provider="ollama"}', "Ollama avg"),
        ('rag_embedding_texts_count_sum{provider="openai"} / rag_embedding_texts_count_count{provider="openai"}', "OpenAI avg"),
    ], 16, y, w=8))
    y += 8

    panels.append(stat_panel("Ollama Success", 'sum(rag_embedding_requests_total{provider="ollama",status="success"})', 0, y, w=4, color="green"))
    panels.append(stat_panel("Ollama Failures", 'sum(rag_embedding_requests_total{provider="ollama",status="failure"})', 4, y, w=4, color="red"))
    panels.append(stat_panel("OpenAI Success", 'sum(rag_embedding_requests_total{provider="openai",status="success"})', 8, y, w=4, color="green"))
    panels.append(stat_panel("OpenAI Failures", 'sum(rag_embedding_requests_total{provider="openai",status="failure"})', 12, y, w=4, color="red"))
    panels.append(stat_panel("Total Texts Embedded", 'sum(rag_embedding_texts_count_sum)', 16, y, w=4, color="blue"))
    panels.append(stat_panel("Pipeline Embed Duration (avg)", 'rag_pipeline_embed_duration_seconds_sum / rag_pipeline_embed_duration_seconds_count', 20, y, w=4, unit="s", color="purple"))
    y += 4

    # ===== ROW 4: Vector Storage (Qdrant) =====
    panels.append(row("Vector Storage (Qdrant)", y)); y += 1

    panels.append(ts("Qdrant Store/Search Ops/min", [
        ('sum(rate(qdrant_operations_total{operation="store"}[5m])) * 60', "Store"),
        ('sum(rate(qdrant_operations_total{operation="search"}[5m])) * 60', "Search"),
    ], 0, y, w=8, unit="ops"))

    panels.append(ts("Qdrant Operation Duration (avg)", [
        ('qdrant_operation_duration_seconds_sum{operation="store"} / qdrant_operation_duration_seconds_count{operation="store"}', "Store avg"),
        ('qdrant_operation_duration_seconds_sum{operation="search"} / qdrant_operation_duration_seconds_count{operation="search"}', "Search avg"),
    ], 8, y, w=8, unit="s"))

    panels.append(ts("Qdrant Search Results Count", [
        ('qdrant_search_results_count_sum / qdrant_search_results_count_count', "Avg results"),
    ], 16, y, w=8))
    y += 8

    panels.append(stat_panel("Store Success", 'sum(qdrant_operations_total{operation="store",status="success"})', 0, y, w=4, color="green"))
    panels.append(stat_panel("Store Failures", 'sum(qdrant_operations_total{operation="store",status="failure"})', 4, y, w=4, color="red"))
    panels.append(stat_panel("Search Success", 'sum(qdrant_operations_total{operation="search",status="success"})', 8, y, w=4, color="green"))
    panels.append(stat_panel("Search Failures", 'sum(qdrant_operations_total{operation="search",status="failure"})', 12, y, w=4, color="red"))
    panels.append(stat_panel("Health OK", 'sum(qdrant_health_checks_total{status="success"})', 16, y, w=4, color="green"))
    panels.append(stat_panel("Health Fail", 'sum(qdrant_health_checks_total{status="failure"})', 20, y, w=4, color="red"))
    y += 4

    # ===== ROW 5: Semantic Retrieval =====
    panels.append(row("Semantic Retrieval", y)); y += 1

    panels.append(ts("Retrieval Rate/min by Strategy", [
        ('sum(rate(rag_retrieval_total{strategy="vector"}[5m])) * 60', "Vector"),
        ('sum(rate(rag_retrieval_total{strategy="mmr"}[5m])) * 60', "MMR"),
    ], 0, y, w=8, unit="ops"))

    panels.append(ts("Retrieval Duration by Strategy", [
        ('rag_retrieval_duration_seconds_sum{strategy="vector"} / rag_retrieval_duration_seconds_count{strategy="vector"}', "Vector avg"),
        ('rag_retrieval_duration_seconds_sum{strategy="mmr"} / rag_retrieval_duration_seconds_count{strategy="mmr"}', "MMR avg"),
    ], 8, y, w=8, unit="s"))

    panels.append(ts("Results Count by Strategy", [
        ('rag_retrieval_results_count_sum{strategy="vector"} / rag_retrieval_results_count_count{strategy="vector"}', "Vector avg"),
        ('rag_retrieval_results_count_sum{strategy="mmr"} / rag_retrieval_results_count_count{strategy="mmr"}', "MMR avg"),
    ], 16, y, w=8))
    y += 8

    panels.append(stat_panel("Vector Retrievals", 'sum(rag_retrieval_total{strategy="vector",status="success"})', 0, y, w=4, color="green"))
    panels.append(stat_panel("MMR Retrievals", 'sum(rag_retrieval_total{strategy="mmr",status="success"})', 4, y, w=4, color="purple"))
    panels.append(stat_panel("Retrieval Failures", 'sum(rag_retrieval_total{status="failure"})', 8, y, w=4, color="red"))
    panels.append(stat_panel("MMR Candidates (avg)", 'rag_retrieval_candidates_count_sum / rag_retrieval_candidates_count_count', 12, y, w=4, color="orange"))
    panels.append(stat_panel("Query Top Score (avg)", 'rag_pipeline_query_top_score_sum / rag_pipeline_query_top_score_count', 16, y, w=4, unit="short", color="blue"))
    panels.append(stat_panel("Query Results (avg)", 'rag_pipeline_query_results_count_sum / rag_pipeline_query_results_count_count', 20, y, w=4, color="green"))
    y += 4

    # ===== ROW 6: Context Assembly =====
    panels.append(row("Context Assembly", y)); y += 1

    panels.append(ts("Context Assembly Rate/min", [
        ('sum(rate(rag_context_assembly_total[5m])) * 60', "Assemblies/min"),
    ], 0, y, w=8, unit="ops"))

    panels.append(ts("Chunks Included vs Skipped (avg)", [
        ('rag_context_chunks_included_sum / rag_context_chunks_included_count', "Included avg"),
        ('rag_context_chunks_skipped_sum / rag_context_chunks_skipped_count', "Skipped avg"),
    ], 8, y, w=8))

    panels.append(ts("Context Length (chars avg)", [
        ('rag_context_length_chars_sum / rag_context_length_chars_count', "Avg length"),
    ], 16, y, w=8, unit="short"))
    y += 8

    panels.append(stat_panel("Total Assemblies", 'sum(rag_context_assembly_total)', 0, y, w=6, color="blue"))
    panels.append(stat_panel("Avg Chunks Included", 'rag_context_chunks_included_sum / rag_context_chunks_included_count', 6, y, w=6, color="green"))
    panels.append(stat_panel("Total Chunks Skipped", 'sum(rag_context_chunks_skipped_sum)', 12, y, w=6, color="yellow"))
    panels.append(stat_panel("Avg Context Length", 'rag_context_length_chars_sum / rag_context_length_chars_count', 18, y, w=6, unit="short", color="purple"))
    y += 4

    # ===== ROW 7: Pipeline Orchestrator Breakdown =====
    panels.append(row("RAG Pipeline Orchestrator (Stage Breakdown)", y)); y += 1

    panels.append(ts("Ingest Stage Durations (avg)", [
        ('rag_pipeline_load_duration_seconds_sum / rag_pipeline_load_duration_seconds_count', "Load"),
        ('rag_pipeline_chunk_duration_seconds_sum / rag_pipeline_chunk_duration_seconds_count', "Chunk"),
        ('rag_pipeline_embed_duration_seconds_sum / rag_pipeline_embed_duration_seconds_count', "Embed"),
        ('rag_pipeline_store_duration_seconds_sum / rag_pipeline_store_duration_seconds_count', "Store"),
    ], 0, y, w=12, unit="s"))

    panels.append(ts("End-to-End Ingest & Query Duration", [
        ('rag_pipeline_ingest_duration_seconds_sum / rag_pipeline_ingest_duration_seconds_count', "Ingest avg"),
        ('rag_pipeline_query_duration_seconds_sum / rag_pipeline_query_duration_seconds_count', "Query avg"),
    ], 12, y, w=12, unit="s"))
    y += 8

    panels.append(stat_panel("Avg Chunks/Doc", 'rag_pipeline_chunks_per_document_sum / rag_pipeline_chunks_per_document_count', 0, y, w=4, color="blue"))
    panels.append(stat_panel("RAG Store Chunks", 'sum(rag_store_chunks_total)', 4, y, w=4, color="green"))
    panels.append(stat_panel("RAG Store Searches", 'sum(rag_store_search_total)', 8, y, w=4, color="purple"))
    panels.append(stat_panel("RAG Store Deletes", 'sum(rag_store_delete_total)', 12, y, w=4, color="orange"))
    panels.append(stat_panel("Store Duration (avg)", 'rag_store_duration_seconds_sum / rag_store_duration_seconds_count', 16, y, w=4, unit="s", color="blue"))
    panels.append(stat_panel("Store Errors", 'sum(rag_store_errors_total)', 20, y, w=4, color="red"))
    y += 4

    # ===== ROW 8: RAG Logs =====
    panels.append(row("RAG Pipeline Logs", y)); y += 1

    panels.append(logs_panel(
        "RAG Events (Loki)",
        '{service=~"api|worker"} |= "rag_data"',
        0, y, w=24, h=10,
    ))
    y += 10

    dashboard = {
        "uid": "ia-agent-fwk-rag-pipeline",
        "title": "IA Agent FWK - RAG Pipeline",
        "tags": ["ia-agent-fwk", "rag", "pipeline", "embeddings", "retrieval"],
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
    out = "docker/grafana/dashboards/rag-pipeline.json"
    with open(out, "w") as f:
        json.dump(dashboard, f, indent=2)
    print(f"\nDashboard written to {out}")
