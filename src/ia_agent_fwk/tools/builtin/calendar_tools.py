"""Calendar agent built-in tools.

Provides tools for email parsing, LLM-based event extraction, event
validation, duplicate checking, and Google Calendar integration.
All tools follow the framework's ``Tool`` ABC pattern.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import date as date_type
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from ia_agent_fwk.tools.base import Tool
from ia_agent_fwk.tools.builtin.calendar_models import CalendarAgentStore
from ia_agent_fwk.tools.exceptions import ToolExecutionError

if TYPE_CHECKING:
    from pathlib import Path

    from ia_agent_fwk.llm.base import LLMProvider
    from ia_agent_fwk.tools.base import ToolContext

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: optional html2text import
# ---------------------------------------------------------------------------


def _html_to_text(html: str) -> str:
    """Convert HTML to plain text using ``html2text`` if available."""
    try:
        import html2text  # noqa: PLC0415

        converter = html2text.HTML2Text()
        converter.ignore_links = False
        converter.ignore_images = True
        converter.body_width = 0
        return converter.handle(html)
    except ImportError:
        # Fallback: strip tags with regex
        text = re.sub(r"<[^>]+>", " ", html)
        return re.sub(r"\s+", " ", text).strip()


# ===========================================================================
# EmailParserTool
# ===========================================================================


class EmailParserInput(BaseModel):
    """Input schema for the email parser tool."""

    model_config = ConfigDict(frozen=True)

    raw_text: str = Field(description="Raw email body text (HTML or plain text)")
    raw_html: str = Field(default="", description="Raw HTML body if available")


class EmailParserOutput(BaseModel):
    """Output schema for the email parser tool."""

    model_config = ConfigDict(frozen=True)

    clean_text: str
    subject: str = ""
    forwarded_from: str = ""
    is_forwarded: bool = False


# Forwarding header patterns
_GMAIL_FWD_RE = re.compile(r"-{5,}\s*Forwarded message\s*-{5,}", re.IGNORECASE)
_OUTLOOK_FWD_RE = re.compile(r"From:\s*[^\n]+\nSent:\s*[^\n]+\nTo:\s*[^\n]+\nSubject:\s*[^\n]+", re.IGNORECASE)
_FROM_RE = re.compile(r"From:\s*(.+?)(?:\n|$)", re.IGNORECASE)
_SUBJECT_RE = re.compile(r"Subject:\s*(.+?)(?:\n|$)", re.IGNORECASE)

# Signature patterns
_SIGNATURE_MARKERS = [
    "\n-- \n",
    "\n--\n",
    "\nSent from my ",
    "\nGet Outlook for ",
    "\n_______",
]

# Quoted thread patterns
_QUOTED_RE = re.compile(r"^>.*$", re.MULTILINE)
_ON_WROTE_RE = re.compile(r"\nOn .+ wrote:\s*$", re.MULTILINE)


class EmailParserTool(Tool):
    """Parse and normalize a forwarded email.

    Strips signatures, quoted threads, and HTML formatting.
    Detects Gmail and Outlook forwarding patterns.
    """

    @property
    def name(self) -> str:
        return "email_parser"

    @property
    def description(self) -> str:
        return "Parse and normalize a forwarded email, stripping signatures and HTML."

    @property
    def input_schema(self) -> type[BaseModel]:
        return EmailParserInput

    @property
    def output_schema(self) -> type[BaseModel]:
        return EmailParserOutput

    @property
    def tags(self) -> list[str]:
        return ["calendar", "email", "parsing"]

    async def execute(self, validated_input: BaseModel, _context: ToolContext) -> BaseModel:
        inp: EmailParserInput = validated_input  # type: ignore[assignment]

        # Choose best source
        text = _html_to_text(inp.raw_html) if inp.raw_html else inp.raw_text

        # Detect forwarding
        is_forwarded = False
        forwarded_from = ""
        subject = ""

        if _GMAIL_FWD_RE.search(text):
            is_forwarded = True
            # Extract from/subject from forwarding header
            from_match = _FROM_RE.search(text)
            if from_match:
                forwarded_from = from_match.group(1).strip()
            subject_match = _SUBJECT_RE.search(text)
            if subject_match:
                subject = subject_match.group(1).strip()
            # Remove the forwarding header block
            text = _GMAIL_FWD_RE.split(text, maxsplit=1)[-1]
        elif _OUTLOOK_FWD_RE.search(text):
            is_forwarded = True
            from_match = _FROM_RE.search(text)
            if from_match:
                forwarded_from = from_match.group(1).strip()
            subject_match = _SUBJECT_RE.search(text)
            if subject_match:
                subject = subject_match.group(1).strip()

        # Strip signatures
        for marker in _SIGNATURE_MARKERS:
            idx = text.find(marker)
            if idx != -1:
                text = text[:idx]

        # Strip quoted threads
        text = _ON_WROTE_RE.sub("", text)
        text = _QUOTED_RE.sub("", text)

        # Normalize whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)
        clean_text = text.strip()

        return EmailParserOutput(
            clean_text=clean_text,
            subject=subject,
            forwarded_from=forwarded_from,
            is_forwarded=is_forwarded,
        )


# ===========================================================================
# EventExtractorTool
# ===========================================================================


_EXTRACTION_SYSTEM_PROMPT = """\
You are an event extraction assistant. Analyze the email text and determine:
1. Whether it contains a calendar event (EVENT_FOUND, NO_EVENT, or UNCERTAIN)
2. If EVENT_FOUND or UNCERTAIN, extract the structured event details

Respond ONLY with a JSON object matching this schema:
{
  "detection_status": "EVENT_FOUND or NO_EVENT or UNCERTAIN",
  "title": "string",
  "date": "YYYY-MM-DD",
  "start_time": "HH:mm",
  "end_time": "HH:mm",
  "duration_minutes": integer,
  "location": "string",
  "meeting_link": "string or empty",
  "participants": ["email@example.com"],
  "description": "brief description",
  "confidence": 0.0 to 1.0
}

If NO_EVENT, return: {"detection_status": "NO_EVENT", "confidence": 1.0}
If fields are unknown, use empty strings or 0. Always include confidence.
Do NOT include any text outside the JSON object.
"""


class EventExtractorInput(BaseModel):
    """Input schema for the event extractor tool."""

    model_config = ConfigDict(frozen=True)

    clean_email_text: str = Field(description="Cleaned email text to analyze for events")
    corrections_context: str = Field(default="", description="Past corrections as few-shot context")


class EventExtractorOutput(BaseModel):
    """Output schema for the event extractor tool."""

    model_config = ConfigDict(frozen=True)

    detection_status: str = "NO_EVENT"
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
    raw_llm_response: str = ""


class EventExtractorTool(Tool):
    """Use LLM to detect and extract calendar events from email text.

    Calls the configured ``LLMProvider`` directly to analyze the email
    content and return structured event data as JSON.
    """

    def __init__(self, provider: LLMProvider) -> None:
        self._provider = provider

    @property
    def name(self) -> str:
        return "event_extractor"

    @property
    def description(self) -> str:
        return "Detect and extract calendar events from cleaned email text using LLM."

    @property
    def input_schema(self) -> type[BaseModel]:
        return EventExtractorInput

    @property
    def output_schema(self) -> type[BaseModel]:
        return EventExtractorOutput

    @property
    def tags(self) -> list[str]:
        return ["calendar", "llm", "extraction"]

    async def execute(self, validated_input: BaseModel, _context: ToolContext) -> BaseModel:
        from ia_agent_fwk.llm.models import Message  # noqa: PLC0415

        inp: EventExtractorInput = validated_input  # type: ignore[assignment]

        system_prompt = _EXTRACTION_SYSTEM_PROMPT
        if inp.corrections_context:
            system_prompt += "\n\nPast corrections from the user (learn from these):\n" + inp.corrections_context

        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=f"Extract event from this email:\n\n{inp.clean_email_text}"),
        ]

        max_retries = 2
        raw_response = ""
        for attempt in range(max_retries + 1):
            chat_resp = await self._provider.chat(messages)
            raw_response = chat_resp.message.content or ""

            parsed = self._try_parse_json(raw_response)
            if parsed is not None:
                try:
                    duration = int(parsed.get("duration_minutes", 30))
                except (ValueError, TypeError):
                    duration = 30
                try:
                    confidence = float(parsed.get("confidence", 0.0))
                except (ValueError, TypeError):
                    confidence = 0.0

                return EventExtractorOutput(
                    detection_status=str(parsed.get("detection_status", "NO_EVENT")),
                    title=str(parsed.get("title", "")),
                    date=str(parsed.get("date", "")),
                    start_time=str(parsed.get("start_time", "")),
                    end_time=str(parsed.get("end_time", "")),
                    duration_minutes=duration,
                    location=str(parsed.get("location", "")),
                    meeting_link=str(parsed.get("meeting_link", "")),
                    participants=list(parsed.get("participants", [])),
                    description=str(parsed.get("description", "")),
                    confidence=confidence,
                    raw_llm_response=raw_response,
                )

            if attempt < max_retries:
                messages.append(Message(role="assistant", content=raw_response))
                messages.append(
                    Message(
                        role="user",
                        content="Your response was not valid JSON. Please respond with ONLY a JSON object.",
                    )
                )
                logger.warning("Event extraction JSON parse failed (attempt %d), retrying", attempt + 1)

        # All retries exhausted — return NO_EVENT
        logger.warning("Event extraction failed after %d attempts", max_retries + 1)
        return EventExtractorOutput(
            detection_status="NO_EVENT",
            confidence=0.0,
            raw_llm_response=raw_response,
        )

    @staticmethod
    def _try_parse_json(text: str) -> dict[str, Any] | None:
        """Try to parse JSON from text, handling markdown code fences."""
        text = text.strip()
        # Remove markdown code fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
            text = text.strip()
        try:
            result: dict[str, Any] = json.loads(text)
        except json.JSONDecodeError:
            return None
        else:
            return result if isinstance(result, dict) else None


# ===========================================================================
# EventValidatorTool
# ===========================================================================


class EventValidatorInput(BaseModel):
    """Input schema for the event validator tool."""

    model_config = ConfigDict(frozen=True)

    title: str = Field(description="Event title")
    date: str = Field(description="Event date YYYY-MM-DD")
    start_time: str = Field(description="Start time HH:mm")
    end_time: str = Field(default="", description="End time HH:mm")
    duration_minutes: int = Field(default=30, description="Duration in minutes")
    confidence: float = Field(default=0.0, description="Extraction confidence 0.0-1.0")
    timezone: str = Field(default="UTC", description="User timezone")
    default_duration: int = Field(default=30, description="Default duration if missing")
    allow_past_dates: bool = Field(default=False, description="Allow past dates")


class EventValidatorOutput(BaseModel):
    """Output schema for the event validator tool."""

    model_config = ConfigDict(frozen=True)

    is_valid: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    corrected_end_time: str = ""
    corrected_duration_minutes: int = 30
    needs_confirmation: bool = False


class EventValidatorTool(Tool):
    """Validate extracted event data.

    Checks date, time, duration, and confidence thresholds.
    Calculates missing end_time from start_time + duration.
    """

    @property
    def name(self) -> str:
        return "event_validator"

    @property
    def description(self) -> str:
        return "Validate extracted event data (dates, times, duration, confidence)."

    @property
    def input_schema(self) -> type[BaseModel]:
        return EventValidatorInput

    @property
    def output_schema(self) -> type[BaseModel]:
        return EventValidatorOutput

    @property
    def tags(self) -> list[str]:
        return ["calendar", "validation"]

    async def execute(self, validated_input: BaseModel, _context: ToolContext) -> BaseModel:  # noqa: C901, PLR0912
        inp: EventValidatorInput = validated_input  # type: ignore[assignment]

        errors: list[str] = []
        warnings: list[str] = []

        # Validate title
        if not inp.title.strip():
            errors.append("Title is empty")

        # Validate date
        event_date: date_type | None = None
        if not inp.date:
            errors.append("Date is missing")
        else:
            try:
                event_date = datetime.strptime(inp.date, "%Y-%m-%d").replace(tzinfo=timezone.utc).date()  # noqa: UP017
            except ValueError:
                errors.append(f"Invalid date format: {inp.date} (expected YYYY-MM-DD)")

        if event_date is not None and not inp.allow_past_dates:
            today = datetime.now(tz=timezone.utc).date()  # noqa: UP017
            if event_date < today:
                errors.append(f"Date {inp.date} is in the past")

        # Validate start_time
        start_dt: datetime | None = None
        if not inp.start_time:
            errors.append("Start time is missing")
        else:
            try:
                start_dt = datetime.strptime(inp.start_time, "%H:%M").replace(tzinfo=timezone.utc)  # noqa: UP017
            except ValueError:
                errors.append(f"Invalid start_time format: {inp.start_time} (expected HH:mm)")

        # Calculate end_time and duration
        duration = inp.duration_minutes if inp.duration_minutes > 0 else inp.default_duration
        corrected_end_time = inp.end_time

        if start_dt is not None:
            if inp.end_time:
                try:
                    end_dt = datetime.strptime(inp.end_time, "%H:%M").replace(tzinfo=timezone.utc)  # noqa: UP017
                    diff = (end_dt - start_dt).total_seconds() / 60
                    if diff > 0:
                        duration = int(diff)
                    else:
                        warnings.append("end_time is before start_time — using duration instead")
                        end_dt = start_dt + timedelta(minutes=duration)
                        corrected_end_time = end_dt.strftime("%H:%M")
                except ValueError:
                    warnings.append(f"Invalid end_time format: {inp.end_time} — calculating from duration")
                    end_dt = start_dt + timedelta(minutes=duration)
                    corrected_end_time = end_dt.strftime("%H:%M")
            else:
                end_dt = start_dt + timedelta(minutes=duration)
                corrected_end_time = end_dt.strftime("%H:%M")

        # Confidence check
        needs_confirmation = inp.confidence < 0.6  # noqa: PLR2004

        return EventValidatorOutput(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            corrected_end_time=corrected_end_time,
            corrected_duration_minutes=duration,
            needs_confirmation=needs_confirmation,
        )


# ===========================================================================
# DuplicateCheckerTool
# ===========================================================================


class DuplicateCheckerInput(BaseModel):
    """Input schema for the duplicate checker tool."""

    model_config = ConfigDict(frozen=True)

    message_id: str = Field(description="Email Message-ID header")
    title: str = Field(description="Event title")
    date: str = Field(description="Event date YYYY-MM-DD")
    start_time: str = Field(description="Event start time HH:mm")


class DuplicateCheckerOutput(BaseModel):
    """Output schema for the duplicate checker tool."""

    model_config = ConfigDict(frozen=True)

    is_duplicate: bool
    duplicate_reason: str = ""


class DuplicateCheckerTool(Tool):
    """Check for duplicate events using email message ID and event hash."""

    def __init__(self, store: CalendarAgentStore) -> None:
        self._store = store

    @property
    def name(self) -> str:
        return "duplicate_checker"

    @property
    def description(self) -> str:
        return "Check if an event already exists (by email ID or event hash)."

    @property
    def input_schema(self) -> type[BaseModel]:
        return DuplicateCheckerInput

    @property
    def output_schema(self) -> type[BaseModel]:
        return DuplicateCheckerOutput

    @property
    def tags(self) -> list[str]:
        return ["calendar", "deduplication"]

    async def execute(self, validated_input: BaseModel, _context: ToolContext) -> BaseModel:
        inp: DuplicateCheckerInput = validated_input  # type: ignore[assignment]

        event_hash = CalendarAgentStore.compute_event_hash(inp.title, inp.date, inp.start_time)

        if inp.message_id and inp.message_id in self._store.processed_emails:
            return DuplicateCheckerOutput(is_duplicate=True, duplicate_reason="message_id")

        if event_hash in self._store.created_events:
            return DuplicateCheckerOutput(is_duplicate=True, duplicate_reason="event_hash")

        return DuplicateCheckerOutput(is_duplicate=False)


# ===========================================================================
# GoogleCalendarTool
# ===========================================================================


def _has_google_libs() -> bool:
    """Check whether Google API client libraries are importable."""
    try:
        import google.auth  # noqa: F401, PLC0415
        import googleapiclient.discovery  # noqa: F401, PLC0415

    except ImportError:
        return False
    else:
        return True


class GoogleCalendarInput(BaseModel):
    """Input schema for the Google Calendar tool."""

    model_config = ConfigDict(frozen=True)

    action: str = Field(description="Action: create, update, or delete")
    title: str = Field(default="", description="Event title")
    description: str = Field(default="", description="Event description")
    date: str = Field(default="", description="Event date YYYY-MM-DD")
    start_time: str = Field(default="", description="Start time HH:mm")
    end_time: str = Field(default="", description="End time HH:mm")
    timezone: str = Field(default="UTC", description="Timezone (e.g. America/Mexico_City)")
    location: str = Field(default="", description="Event location")
    meeting_link: str = Field(default="", description="Meeting URL")
    attendees: list[str] = Field(default_factory=list, description="Attendee email addresses")
    event_id: str = Field(default="", description="Google Calendar event ID (for update/delete)")


class GoogleCalendarOutput(BaseModel):
    """Output schema for the Google Calendar tool."""

    model_config = ConfigDict(frozen=True)

    success: bool
    event_id: str = ""
    event_link: str = ""
    error: str = ""


class GoogleCalendarTool(Tool):
    """Create, update, or delete Google Calendar events via the Google Calendar API.

    Requires OAuth2 credentials configured via ``credentials.json`` and
    ``token.json`` files (see ``setup_oauth.py``).
    """

    def __init__(
        self,
        credentials_path: Path | None = None,
        token_path: Path | None = None,
        calendar_id: str = "primary",
    ) -> None:
        self._credentials_path = credentials_path
        self._token_path = token_path
        self._calendar_id = calendar_id
        self._service: Any = None

    @property
    def name(self) -> str:
        return "google_calendar"

    @property
    def description(self) -> str:
        return "Create, update, or delete events in Google Calendar."

    @property
    def input_schema(self) -> type[BaseModel]:
        return GoogleCalendarInput

    @property
    def output_schema(self) -> type[BaseModel]:
        return GoogleCalendarOutput

    @property
    def tags(self) -> list[str]:
        return ["calendar", "google", "api"]

    async def execute(self, validated_input: BaseModel, _context: ToolContext) -> BaseModel:
        inp: GoogleCalendarInput = validated_input  # type: ignore[assignment]

        if not _has_google_libs():
            return GoogleCalendarOutput(
                success=False,
                error="Google API libraries not installed. Run: pip install ia-agent-fwk[calendar]",
            )

        try:
            service = self._get_service()
        except Exception as exc:  # noqa: BLE001
            return GoogleCalendarOutput(success=False, error=f"Failed to initialize Calendar API: {exc}")

        if inp.action == "create":
            return await self._create_event(service, inp)
        if inp.action == "update":
            return await self._update_event(service, inp)
        if inp.action == "delete":
            return await self._delete_event(service, inp)

        return GoogleCalendarOutput(success=False, error=f"Unknown action: {inp.action}")

    def _get_service(self) -> Any:
        """Lazily initialize the Google Calendar API service."""
        if self._service is not None:
            return self._service

        from google.auth.transport.requests import Request  # noqa: PLC0415
        from google.oauth2.credentials import Credentials  # noqa: PLC0415
        from googleapiclient.discovery import build  # noqa: PLC0415

        creds: Credentials | None = None
        if self._token_path and self._token_path.exists():
            creds = Credentials.from_authorized_user_file(
                str(self._token_path),
                scopes=["https://www.googleapis.com/auth/calendar"],
            )

        if creds is None:
            msg = f"No token.json found at {self._token_path}. Run setup_oauth.py first."
            raise ToolExecutionError(msg)

        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
            if self._token_path:
                self._token_path.write_text(creds.to_json())

        self._service = build("calendar", "v3", credentials=creds)
        return self._service

    async def _create_event(self, service: Any, inp: GoogleCalendarInput) -> GoogleCalendarOutput:
        """Create a new calendar event."""
        event_body = self._build_event_body(inp)
        try:
            event = service.events().insert(calendarId=self._calendar_id, body=event_body).execute()
            return GoogleCalendarOutput(
                success=True,
                event_id=event.get("id", ""),
                event_link=event.get("htmlLink", ""),
            )
        except Exception as exc:  # noqa: BLE001
            return GoogleCalendarOutput(success=False, error=f"Calendar API error: {exc}")

    async def _update_event(self, service: Any, inp: GoogleCalendarInput) -> GoogleCalendarOutput:
        """Update an existing calendar event."""
        if not inp.event_id:
            return GoogleCalendarOutput(success=False, error="event_id is required for update")
        event_body = self._build_event_body(inp)
        try:
            event = (
                service.events().update(calendarId=self._calendar_id, eventId=inp.event_id, body=event_body).execute()
            )
            return GoogleCalendarOutput(
                success=True,
                event_id=event.get("id", ""),
                event_link=event.get("htmlLink", ""),
            )
        except Exception as exc:  # noqa: BLE001
            return GoogleCalendarOutput(success=False, error=f"Calendar API error: {exc}")

    async def _delete_event(self, service: Any, inp: GoogleCalendarInput) -> GoogleCalendarOutput:
        """Delete a calendar event."""
        if not inp.event_id:
            return GoogleCalendarOutput(success=False, error="event_id is required for delete")
        try:
            service.events().delete(calendarId=self._calendar_id, eventId=inp.event_id).execute()
            return GoogleCalendarOutput(success=True, event_id=inp.event_id)
        except Exception as exc:  # noqa: BLE001
            return GoogleCalendarOutput(success=False, error=f"Calendar API error: {exc}")

    @staticmethod
    def _build_event_body(inp: GoogleCalendarInput) -> dict[str, Any]:
        """Build a Google Calendar API event body."""
        body: dict[str, Any] = {
            "summary": inp.title,
            "description": inp.description,
        }

        if inp.date and inp.start_time:
            start_str = f"{inp.date}T{inp.start_time}:00"
            body["start"] = {"dateTime": start_str, "timeZone": inp.timezone}
            end_str = f"{inp.date}T{inp.end_time}:00" if inp.end_time else start_str
            body["end"] = {"dateTime": end_str, "timeZone": inp.timezone}

        if inp.location:
            body["location"] = inp.location

        if inp.attendees:
            body["attendees"] = [{"email": e} for e in inp.attendees]

        if inp.meeting_link:
            body["description"] = f"{inp.description}\n\nMeeting link: {inp.meeting_link}".strip()

        return body
