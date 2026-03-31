"""Context processing — clean RAG context and inject into LLM prompts."""

from __future__ import annotations

import re


def clean_context(context: str, cleanup_prefixes: list[str]) -> str:
    """Remove contextual enrichment prefixes from RAG chunks.

    The prefixes are generated during ingestion by contextual enrichment
    and confuse the LLM if left in. Prefix list comes from config.
    """
    if not context:
        return ""

    for prefix in cleanup_prefixes:
        escaped = re.escape(prefix.rstrip())
        context = re.sub(
            escaped + r"\s*\n\n[^\n]+(?:palabras clave[^\n]*)?\n\n",
            "",
            context,
        )
        context = re.sub(escaped + r"\s*\n\n[^\n]+\n\n", "", context)
        context = context.replace(prefix, "")

    return context.strip()


def inject_context(system_prompt: str, context: str, template: str) -> str:
    """Append RAG context to system prompt using the configured template.

    Template must contain {context} placeholder.
    """
    if not context:
        return system_prompt
    return system_prompt + template.format(context=context)
