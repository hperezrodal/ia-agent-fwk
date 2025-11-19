"""Agent execution endpoints."""

from __future__ import annotations

import logging
import time
from typing import Annotated

from fastapi import APIRouter, Depends, Request, Security

from ia_agent_fwk.agents.config import AgentConfig
from ia_agent_fwk.agents.factory import AgentFactory
from ia_agent_fwk.agents.registry import AgentRegistry
from ia_agent_fwk.api.dependencies import (
    check_rate_limit,
    get_conversation_backend,
    get_memory_backend,
    get_settings,
    require_api_key,
)
from ia_agent_fwk.api.models import AgentRunRequest, AgentRunResponse, TokenUsageResponse
from ia_agent_fwk.config.settings import AppSettings  # noqa: TC001
from ia_agent_fwk.memory.backends.conversation import ConversationMemoryBackend  # noqa: TC001
from ia_agent_fwk.memory.base import MemoryBackend  # noqa: TC001
from ia_agent_fwk.observability.metrics import get_metrics_collector
from ia_agent_fwk.observability.tracing import get_tracer
from ia_agent_fwk.security.audit import AuditLogger  # noqa: TC001

logger = logging.getLogger(__name__)
_tracer = get_tracer(__name__)

router = APIRouter(
    prefix="/api/v1",
    tags=["agents"],
    dependencies=[Security(require_api_key), Depends(check_rate_limit)],
)

# Default system prompts for example agent types
# Tools allowed per agent type (None = all tools)
_AGENT_TOOLS: dict[str, list[str] | None] = {
    "customer_support": [
        "ticket_lookup",
        "faq_search",
        "escalation",
        "response_draft",
    ],
    "document_processor": [
        "rag_search",
        "list_documents",
        "load_document",
        "text_extractor",
        "section_identifier",
        "entity_extractor",
        "summarizer",
    ],
    "finance": [
        "financial_data_lookup",
        "ratio_calculator",
        "anomaly_detector",
        "report_generator",
        "calculator",
    ],
}

# Default system prompts for example agent types
_AGENT_SYSTEM_PROMPTS: dict[str, str] = {
    "customer_support": (
        "You are a helpful customer support agent. You have access to tools for "
        "looking up support tickets (ticket_lookup), searching FAQs (faq_search), "
        "escalating issues (escalation), and drafting responses (response_draft). "
        "Always use the available tools to provide accurate information. "
        "Be polite, concise, and helpful."
    ),
    "document_processor": (
        "You are a document analysis agent. You MUST use your tools to answer every question.\n\n"
        "MANDATORY WORKFLOW for every user question:\n"
        "1. ALWAYS call 'rag_search' first with the user's question to find relevant document chunks\n"
        "2. If rag_search returns no results, fall back to 'list_documents' + 'load_document'\n"
        "3. Answer based ONLY on the document content you found\n\n"
        "NEVER answer from your own knowledge. NEVER say you don't have information "
        "without first searching documents. You MUST call tools proactively.\n\n"
        "Available tools:\n"
        "- rag_search: Semantic search across all ingested documents (USE THIS FIRST)\n"
        "- list_documents: List all available document files\n"
        "- load_document: Load full content of a specific document\n"
        "- entity_extractor: Extract entities from text\n"
        "- summarizer: Summarize document content\n\n"
        "Answer in the same language as the user's question."
    ),
    "finance": (
        "You are a financial analysis agent. You have access to tools for "
        "looking up financial data (financial_data_lookup), calculating financial "
        "ratios (ratio_calculator), detecting anomalies (anomaly_detector), "
        "and generating reports (report_generator). "
        "Always use the available tools to provide data-driven analysis. "
        "Be precise with numbers and cite your sources."
    ),
}


@router.post("/agents/{agent_type}/run", response_model=AgentRunResponse)
async def run_agent(  # noqa: PLR0913
    agent_type: str,
    request_body: AgentRunRequest,
    request: Request,
    settings: Annotated[AppSettings, Depends(get_settings)],
    conversation_backend: Annotated[ConversationMemoryBackend, Depends(get_conversation_backend)],
    memory_backend: Annotated[MemoryBackend, Depends(get_memory_backend)],
) -> AgentRunResponse:
    """Execute an agent synchronously."""
    collector = get_metrics_collector()
    start = time.monotonic()

    # Validate agent_type against registry (raises AgentConfigError -> 404)
    AgentRegistry.get(agent_type)

    # Resolve conversation (CRUD concern stays in route)
    conversation_id = request_body.conversation_id
    if conversation_id is None:
        conv_info = await conversation_backend.create_conversation(
            agent_namespace=agent_type,
        )
        conversation_id = conv_info.conversation_id
    else:
        existing = await conversation_backend.get_conversation(conversation_id)
        if existing is None:
            conv_info = await conversation_backend.create_conversation(
                agent_namespace=agent_type,
                conversation_id=conversation_id,
            )
            conversation_id = conv_info.conversation_id

    # Create agent with type-specific system prompt, tools, and memory backends
    system_prompt = _AGENT_SYSTEM_PROMPTS.get(agent_type, "")
    agent_config = AgentConfig(
        name=f"{agent_type}-api",
        agent_type=agent_type,
        provider_name=settings.llm.default_provider,
        system_prompt=system_prompt,
    )

    # Build a filtered tool registry if this agent type has a tool allowlist
    allowed_tools = _AGENT_TOOLS.get(agent_type)
    tool_registry = None
    if allowed_tools is not None:
        from ia_agent_fwk.tools.builtin import register_builtin_tools  # noqa: PLC0415
        from ia_agent_fwk.tools.registry import ToolRegistry  # noqa: PLC0415

        full_registry = ToolRegistry()
        register_builtin_tools(full_registry)
        tool_registry = ToolRegistry()
        for tool_name in allowed_tools:
            if full_registry.has(tool_name):
                tool_registry.register(full_registry.get(tool_name))

    agent = AgentFactory.create(
        agent_config,
        settings.llm,
        tool_registry=tool_registry,
        memory_backend=memory_backend,
        conversation_backend=conversation_backend,
    )

    # Execute -- agent handles history loading, memory search, and persistence
    result = await agent.run(
        request_body.prompt,
        conversation_id=conversation_id,
    )

    duration_ms = (time.monotonic() - start) * 1000

    # Metrics
    collector.increment(
        "api_agent_executions_total",
        labels={"agent_type": agent_type, "status": "success"},
    )
    collector.observe("api_agent_execution_duration_seconds", duration_ms / 1000, labels={"agent_type": agent_type})
    collector.observe("api_agent_prompt_tokens", result.usage.prompt_tokens, labels={"agent_type": agent_type})
    collector.observe("api_agent_completion_tokens", result.usage.completion_tokens, labels={"agent_type": agent_type})
    collector.observe("api_agent_iterations", result.iterations, labels={"agent_type": agent_type})

    logger.info(
        "Agent executed: type=%s, conv=%s, tokens=%d/%d, iters=%d (%.1fms)",
        agent_type,
        conversation_id,
        result.usage.prompt_tokens,
        result.usage.completion_tokens,
        result.iterations,
        duration_ms,
        extra={
            "api_data": {
                "event": "agent_executed",
                "agent_type": agent_type,
                "conversation_id": conversation_id,
                "prompt_tokens": result.usage.prompt_tokens,
                "completion_tokens": result.usage.completion_tokens,
                "total_tokens": result.usage.total_tokens,
                "iterations": result.iterations,
                "duration_ms": round(duration_ms, 1),
            }
        },
    )

    # Audit log agent execution
    audit_logger: AuditLogger | None = getattr(request.app.state, "audit_logger", None)
    if audit_logger is not None:
        api_key = request.headers.get("x-api-key", "")
        audit_logger.log_agent_execution(
            api_key=api_key,
            agent_type=agent_type,
            result="success",
            metadata={"iterations": result.iterations, "duration_ms": result.duration_ms},
        )

    return AgentRunResponse(
        conversation_id=conversation_id,
        output=result.output,
        usage=TokenUsageResponse(
            prompt_tokens=result.usage.prompt_tokens,
            completion_tokens=result.usage.completion_tokens,
            total_tokens=result.usage.total_tokens,
        ),
        iterations=result.iterations,
        duration_ms=result.duration_ms,
        agent_type=agent_type,
    )
