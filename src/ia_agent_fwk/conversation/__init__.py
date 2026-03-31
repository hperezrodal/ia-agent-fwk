"""Conversational RAG agent — generic, multi-tenant, config-driven."""

from __future__ import annotations

from ia_agent_fwk.conversation.agent import AgentResponse, ConversationalRAGAgent
from ia_agent_fwk.conversation.classifier import ClassifyResult, MessageClassifier
from ia_agent_fwk.conversation.context import clean_context, inject_context
from ia_agent_fwk.conversation.endpoints import mount_chat_endpoints
from ia_agent_fwk.conversation.session import SessionManager

__all__ = [
    "AgentResponse",
    "ClassifyResult",
    "ConversationalRAGAgent",
    "MessageClassifier",
    "SessionManager",
    "clean_context",
    "inject_context",
    "mount_chat_endpoints",
]
