"""Calendar agent data models and in-memory store.

Provides Pydantic models for extracted events, email records,
calendar event references, user corrections, and pending
confirmations.  ``CalendarAgentStore`` manages these in memory
with JSON file persistence.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class ExtractedEvent(BaseModel):
    """Structured event data from LLM extraction."""

    model_config = ConfigDict(frozen=True)

    title: str = ""
    date: str = ""
    start_time: str = ""
    end_time: str = ""
    duration_minutes: int = 30
    location: str = ""
    meeting_link: str = ""
    participants: list[str] = Field(default_factory=list)
    description: str = ""
    confidence: float = 0.0
    detection_status: str = "NO_EVENT"


class EmailRecord(BaseModel):
    """Tracked processed email."""

    model_config = ConfigDict(frozen=True)

    message_id: str
    subject: str = ""
    sender: str = ""
    received_at: str = ""
    status: Literal["processed", "no_event", "pending_confirmation", "error"] = "processed"
    event_hash: str | None = None


class CalendarEventRef(BaseModel):
    """Reference to a created Google Calendar event."""

    model_config = ConfigDict(frozen=True)

    google_event_id: str
    email_message_id: str
    event_hash: str
    created_at: str = ""
    title: str = ""


class UserCorrection(BaseModel):
    """Stored correction for learning."""

    model_config = ConfigDict(frozen=True)

    email_snippet: str = ""
    original_extraction: dict[str, Any] = Field(default_factory=dict)
    corrected_fields: dict[str, Any] = Field(default_factory=dict)
    created_at: str = ""


class PendingConfirmation(BaseModel):
    """Event awaiting user confirmation."""

    model_config = ConfigDict(frozen=True)

    confirmation_id: str
    email_message_id: str
    extracted_event: dict[str, Any] = Field(default_factory=dict)
    sent_at: str = ""
    status: Literal["pending", "confirmed", "cancelled", "edited"] = "pending"


# ---------------------------------------------------------------------------
# In-memory store with JSON persistence
# ---------------------------------------------------------------------------


class CalendarAgentStore:
    """In-memory store with JSON file persistence."""

    def __init__(self) -> None:
        self.processed_emails: dict[str, EmailRecord] = {}
        self.created_events: dict[str, CalendarEventRef] = {}
        self.corrections: list[UserCorrection] = []
        self.pending_confirmations: dict[str, PendingConfirmation] = {}

    # -- Deduplication ---------------------------------------------------

    @staticmethod
    def compute_event_hash(title: str, date: str, start_time: str) -> str:
        """Compute a deterministic hash for duplicate detection."""
        key = f"{title.lower().strip()}|{date}|{start_time}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    def is_duplicate(self, message_id: str, event_hash: str) -> bool:
        """Return ``True`` if the email or event has already been processed."""
        if message_id and message_id in self.processed_emails:
            return True
        return bool(event_hash and event_hash in self.created_events)

    # -- Processed emails ------------------------------------------------

    def add_processed_email(self, record: EmailRecord) -> None:
        """Track a processed email by its message ID."""
        self.processed_emails[record.message_id] = record

    # -- Created events --------------------------------------------------

    def add_created_event(self, ref: CalendarEventRef) -> None:
        """Track a created calendar event by its event hash."""
        self.created_events[ref.event_hash] = ref

    # -- Corrections -----------------------------------------------------

    def add_correction(
        self,
        correction: UserCorrection,
        max_corrections: int = 20,
    ) -> None:
        """Store a user correction, dropping the oldest if at capacity."""
        self.corrections.append(correction)
        if len(self.corrections) > max_corrections:
            self.corrections = self.corrections[-max_corrections:]

    def get_relevant_corrections(self, limit: int = 5) -> list[UserCorrection]:
        """Return the most recent corrections (simple recency for MVP)."""
        return self.corrections[-limit:]

    # -- Pending confirmations -------------------------------------------

    def add_pending_confirmation(self, confirmation: PendingConfirmation) -> None:
        """Track a pending confirmation request."""
        self.pending_confirmations[confirmation.confirmation_id] = confirmation

    def resolve_confirmation(self, confirmation_id: str, status: str) -> None:
        """Update the status of a pending confirmation."""
        existing = self.pending_confirmations.get(confirmation_id)
        if existing is not None:
            self.pending_confirmations[confirmation_id] = PendingConfirmation(
                confirmation_id=existing.confirmation_id,
                email_message_id=existing.email_message_id,
                extracted_event=existing.extracted_event,
                sent_at=existing.sent_at,
                status=status,
            )

    # -- Persistence -----------------------------------------------------

    def save(self, path: Path) -> None:
        """Serialize the store to a JSON file (atomic write)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "processed_emails": {k: v.model_dump() for k, v in self.processed_emails.items()},
            "created_events": {k: v.model_dump() for k, v in self.created_events.items()},
            "corrections": [c.model_dump() for c in self.corrections],
            "pending_confirmations": {k: v.model_dump() for k, v in self.pending_confirmations.items()},
        }
        tmp_path = path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(data, indent=2, default=str))
        os.replace(tmp_path, path)
        logger.debug("Store saved to %s", path)

    def load(self, path: Path) -> None:
        """Deserialize the store from a JSON file.  No-op if the file is missing."""
        if not path.exists():
            logger.debug("Store file not found at %s — starting fresh", path)
            return
        try:
            raw = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load store from %s: %s", path, exc)
            return

        for k, v in raw.get("processed_emails", {}).items():
            self.processed_emails[k] = EmailRecord(**v)
        for k, v in raw.get("created_events", {}).items():
            self.created_events[k] = CalendarEventRef(**v)
        self.corrections = [UserCorrection(**c) for c in raw.get("corrections", [])]
        for k, v in raw.get("pending_confirmations", {}).items():
            self.pending_confirmations[k] = PendingConfirmation(**v)

        logger.debug(
            "Store loaded from %s: %d emails, %d events, %d corrections",
            path,
            len(self.processed_emails),
            len(self.created_events),
            len(self.corrections),
        )
