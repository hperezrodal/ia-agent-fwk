"""Tests for calendar agent tools."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from ia_agent_fwk.tools.base import ToolContext
from ia_agent_fwk.tools.builtin.calendar_models import (
    CalendarAgentStore,
    CalendarEventRef,
    EmailRecord,
)
from ia_agent_fwk.tools.builtin.calendar_tools import (
    DuplicateCheckerInput,
    DuplicateCheckerTool,
    EmailParserInput,
    EmailParserTool,
    EventExtractorInput,
    EventExtractorOutput,
    EventExtractorTool,
    EventValidatorInput,
    EventValidatorTool,
    GoogleCalendarInput,
    GoogleCalendarTool,
    _html_to_text,
)


@pytest.fixture
def tool_context():
    return ToolContext(execution_id="test-001", agent_id="test-agent")


# ===========================================================================
# _html_to_text helper
# ===========================================================================


@pytest.mark.unit
class TestHtmlToText:
    def test_with_html2text(self):
        html = "<p>Hello <b>world</b></p>"
        result = _html_to_text(html)
        assert "Hello" in result
        assert "world" in result

    def test_fallback_strips_tags(self):
        import sys

        # Temporarily hide html2text so the function uses the regex fallback
        real_mod = sys.modules.get("html2text")
        sys.modules["html2text"] = None  # type: ignore[assignment]
        try:
            # Re-import to clear any cached reference — the function
            # does a local import each time so this is sufficient.
            html = "<p>Hello <b>world</b></p>"
            result = _html_to_text(html)
            assert "<p>" not in result
            assert "Hello" in result
            assert "world" in result
        finally:
            if real_mod is not None:
                sys.modules["html2text"] = real_mod
            else:
                sys.modules.pop("html2text", None)


# ===========================================================================
# EmailParserTool
# ===========================================================================


@pytest.mark.unit
class TestEmailParserTool:
    async def test_properties(self):
        tool = EmailParserTool()
        assert tool.name == "email_parser"
        assert "email" in tool.tags
        assert issubclass(tool.input_schema, BaseModel)
        assert issubclass(tool.output_schema, BaseModel)

    async def test_plain_text_passthrough(self, tool_context):
        tool = EmailParserTool()
        inp = EmailParserInput(raw_text="Hello, meeting tomorrow at 3pm.")
        result = await tool.execute(inp, tool_context)
        assert result.clean_text == "Hello, meeting tomorrow at 3pm."
        assert result.is_forwarded is False

    async def test_gmail_forwarding_detection(self, tool_context):
        tool = EmailParserTool()
        text = (
            "---------- Forwarded message ----------\n"
            "From: Alice <alice@test.com>\n"
            "Subject: Team Meeting\n"
            "Date: Mon, 1 Apr 2026\n\n"
            "Hey, let's meet tomorrow at 10am in Room A."
        )
        inp = EmailParserInput(raw_text=text)
        result = await tool.execute(inp, tool_context)
        assert result.is_forwarded is True
        assert "alice@test.com" in result.forwarded_from
        assert result.subject == "Team Meeting"
        assert "meet tomorrow" in result.clean_text

    async def test_outlook_forwarding_detection(self, tool_context):
        tool = EmailParserTool()
        text = (
            "From: Bob <bob@test.com>\n"
            "Sent: Monday, April 1, 2026 10:00 AM\n"
            "To: Carol <carol@test.com>\n"
            "Subject: Sprint Planning\n\n"
            "Sprint planning at 2pm in the main office."
        )
        inp = EmailParserInput(raw_text=text)
        result = await tool.execute(inp, tool_context)
        assert result.is_forwarded is True
        assert "bob@test.com" in result.forwarded_from
        assert result.subject == "Sprint Planning"

    async def test_signature_stripping(self, tool_context):
        tool = EmailParserTool()
        text = "Meeting at 3pm tomorrow.\n-- \nJohn Doe\nCEO, Acme Corp"
        inp = EmailParserInput(raw_text=text)
        result = await tool.execute(inp, tool_context)
        assert "John Doe" not in result.clean_text
        assert "Meeting at 3pm" in result.clean_text

    async def test_sent_from_signature(self, tool_context):
        tool = EmailParserTool()
        text = "Please join the call at noon.\nSent from my iPhone"
        inp = EmailParserInput(raw_text=text)
        result = await tool.execute(inp, tool_context)
        assert "iPhone" not in result.clean_text

    async def test_quoted_thread_stripping(self, tool_context):
        tool = EmailParserTool()
        text = (
            "Sure, see you at 3pm!\n\n"
            "On Mon, 1 Apr 2026 at 10:00 Alice wrote:\n"
            "> Original message line 1\n"
            "> Original message line 2\n"
        )
        inp = EmailParserInput(raw_text=text)
        result = await tool.execute(inp, tool_context)
        assert "Original message" not in result.clean_text
        assert "see you at 3pm" in result.clean_text

    async def test_html_input_preferred(self, tool_context):
        tool = EmailParserTool()
        inp = EmailParserInput(
            raw_text="Plain text version",
            raw_html="<p>HTML version with <b>bold</b></p>",
        )
        result = await tool.execute(inp, tool_context)
        # When HTML is provided, it should be converted and used
        assert "HTML version" in result.clean_text

    async def test_whitespace_normalization(self, tool_context):
        tool = EmailParserTool()
        text = "Hello\n\n\n\n\nWorld"
        inp = EmailParserInput(raw_text=text)
        result = await tool.execute(inp, tool_context)
        # Should collapse 5+ newlines to 2
        assert "\n\n\n" not in result.clean_text


# ===========================================================================
# EventExtractorTool
# ===========================================================================


def _make_mock_provider(response_content: str) -> MagicMock:
    """Create a mock LLM provider that returns the given content."""
    provider = MagicMock()
    message = MagicMock()
    message.content = response_content
    chat_response = MagicMock()
    chat_response.message = message
    provider.chat = AsyncMock(return_value=chat_response)
    return provider


@pytest.mark.unit
class TestEventExtractorTool:
    async def test_properties(self):
        provider = _make_mock_provider("{}")
        tool = EventExtractorTool(provider)
        assert tool.name == "event_extractor"
        assert "llm" in tool.tags

    async def test_successful_extraction(self, tool_context):
        response_json = json.dumps(
            {
                "detection_status": "EVENT_FOUND",
                "title": "Team Standup",
                "date": "2026-04-01",
                "start_time": "10:00",
                "end_time": "10:30",
                "duration_minutes": 30,
                "location": "Room A",
                "meeting_link": "",
                "participants": ["alice@test.com"],
                "description": "Daily standup",
                "confidence": 0.95,
            }
        )
        provider = _make_mock_provider(response_json)
        tool = EventExtractorTool(provider)
        inp = EventExtractorInput(clean_email_text="Team standup at 10am tomorrow in Room A")
        result = await tool.execute(inp, tool_context)

        assert isinstance(result, EventExtractorOutput)
        assert result.detection_status == "EVENT_FOUND"
        assert result.title == "Team Standup"
        assert result.confidence == 0.95
        assert result.raw_llm_response == response_json

    async def test_no_event_detection(self, tool_context):
        response_json = json.dumps(
            {
                "detection_status": "NO_EVENT",
                "confidence": 1.0,
            }
        )
        provider = _make_mock_provider(response_json)
        tool = EventExtractorTool(provider)
        inp = EventExtractorInput(clean_email_text="Here is the quarterly report.")
        result = await tool.execute(inp, tool_context)

        assert result.detection_status == "NO_EVENT"
        assert result.confidence == 1.0

    async def test_handles_markdown_code_fences(self, tool_context):
        response = '```json\n{"detection_status": "EVENT_FOUND", "title": "Meeting", "confidence": 0.8}\n```'
        provider = _make_mock_provider(response)
        tool = EventExtractorTool(provider)
        inp = EventExtractorInput(clean_email_text="Meeting tomorrow")
        result = await tool.execute(inp, tool_context)

        assert result.detection_status == "EVENT_FOUND"
        assert result.title == "Meeting"

    async def test_retries_on_invalid_json(self, tool_context):
        provider = MagicMock()
        # First call: invalid JSON, second call: valid JSON
        bad_msg = MagicMock()
        bad_msg.content = "This is not JSON"
        bad_response = MagicMock()
        bad_response.message = bad_msg
        good_msg = MagicMock()
        good_msg.content = json.dumps(
            {
                "detection_status": "EVENT_FOUND",
                "title": "Meeting",
                "confidence": 0.7,
            }
        )
        good_response = MagicMock()
        good_response.message = good_msg
        provider.chat = AsyncMock(side_effect=[bad_response, good_response])

        tool = EventExtractorTool(provider)
        inp = EventExtractorInput(clean_email_text="Meeting at 3pm")
        result = await tool.execute(inp, tool_context)

        assert result.detection_status == "EVENT_FOUND"
        assert provider.chat.call_count == 2

    async def test_exhausted_retries_returns_no_event(self, tool_context):
        bad_msg = MagicMock()
        bad_msg.content = "Not JSON at all"
        bad_response = MagicMock()
        bad_response.message = bad_msg
        provider = MagicMock()
        provider.chat = AsyncMock(return_value=bad_response)

        tool = EventExtractorTool(provider)
        inp = EventExtractorInput(clean_email_text="Meeting stuff")
        result = await tool.execute(inp, tool_context)

        assert result.detection_status == "NO_EVENT"
        assert result.confidence == 0.0
        # 1 initial + 2 retries = 3 calls
        assert provider.chat.call_count == 3

    async def test_corrections_context_appended(self, tool_context):
        response_json = json.dumps({"detection_status": "NO_EVENT", "confidence": 1.0})
        provider = _make_mock_provider(response_json)
        tool = EventExtractorTool(provider)

        inp = EventExtractorInput(
            clean_email_text="Some email",
            corrections_context="- Original: 03:00, Corrected: 15:00",
        )
        await tool.execute(inp, tool_context)

        # Verify the system prompt includes corrections
        call_args = provider.chat.call_args
        messages = call_args[0][0]
        system_msg = messages[0].content
        assert "Past corrections" in system_msg
        assert "15:00" in system_msg

    async def test_try_parse_json_non_dict(self):
        result = EventExtractorTool._try_parse_json("[1, 2, 3]")
        assert result is None

    async def test_try_parse_json_valid(self):
        result = EventExtractorTool._try_parse_json('{"key": "value"}')
        assert result == {"key": "value"}


# ===========================================================================
# EventValidatorTool
# ===========================================================================


@pytest.mark.unit
class TestEventValidatorTool:
    async def test_properties(self):
        tool = EventValidatorTool()
        assert tool.name == "event_validator"
        assert "validation" in tool.tags

    async def test_valid_event(self, tool_context):
        tool = EventValidatorTool()
        inp = EventValidatorInput(
            title="Standup",
            date="2027-04-01",
            start_time="10:00",
            end_time="10:30",
            duration_minutes=30,
            confidence=0.9,
            allow_past_dates=False,
        )
        result = await tool.execute(inp, tool_context)
        assert result.is_valid is True
        assert result.errors == []
        assert result.needs_confirmation is False

    async def test_missing_title(self, tool_context):
        tool = EventValidatorTool()
        inp = EventValidatorInput(
            title="",
            date="2027-04-01",
            start_time="10:00",
            confidence=0.9,
        )
        result = await tool.execute(inp, tool_context)
        assert result.is_valid is False
        assert any("Title is empty" in e for e in result.errors)

    async def test_missing_date(self, tool_context):
        tool = EventValidatorTool()
        inp = EventValidatorInput(
            title="Meeting",
            date="",
            start_time="10:00",
            confidence=0.9,
        )
        result = await tool.execute(inp, tool_context)
        assert result.is_valid is False
        assert any("Date is missing" in e for e in result.errors)

    async def test_invalid_date_format(self, tool_context):
        tool = EventValidatorTool()
        inp = EventValidatorInput(
            title="Meeting",
            date="April 1, 2027",
            start_time="10:00",
            confidence=0.9,
        )
        result = await tool.execute(inp, tool_context)
        assert result.is_valid is False
        assert any("Invalid date format" in e for e in result.errors)

    async def test_past_date_rejected(self, tool_context):
        tool = EventValidatorTool()
        inp = EventValidatorInput(
            title="Meeting",
            date="2020-01-01",
            start_time="10:00",
            confidence=0.9,
            allow_past_dates=False,
        )
        result = await tool.execute(inp, tool_context)
        assert result.is_valid is False
        assert any("in the past" in e for e in result.errors)

    async def test_past_date_allowed(self, tool_context):
        tool = EventValidatorTool()
        inp = EventValidatorInput(
            title="Meeting",
            date="2020-01-01",
            start_time="10:00",
            confidence=0.9,
            allow_past_dates=True,
        )
        result = await tool.execute(inp, tool_context)
        assert result.is_valid is True

    async def test_missing_start_time(self, tool_context):
        tool = EventValidatorTool()
        inp = EventValidatorInput(
            title="Meeting",
            date="2027-04-01",
            start_time="",
            confidence=0.9,
        )
        result = await tool.execute(inp, tool_context)
        assert result.is_valid is False
        assert any("Start time is missing" in e for e in result.errors)

    async def test_invalid_start_time_format(self, tool_context):
        tool = EventValidatorTool()
        inp = EventValidatorInput(
            title="Meeting",
            date="2027-04-01",
            start_time="3pm",
            confidence=0.9,
        )
        result = await tool.execute(inp, tool_context)
        assert result.is_valid is False
        assert any("Invalid start_time format" in e for e in result.errors)

    async def test_calculates_end_time_from_duration(self, tool_context):
        tool = EventValidatorTool()
        inp = EventValidatorInput(
            title="Meeting",
            date="2027-04-01",
            start_time="10:00",
            end_time="",
            duration_minutes=60,
            confidence=0.9,
        )
        result = await tool.execute(inp, tool_context)
        assert result.is_valid is True
        assert result.corrected_end_time == "11:00"
        assert result.corrected_duration_minutes == 60

    async def test_end_time_before_start_time_warning(self, tool_context):
        tool = EventValidatorTool()
        inp = EventValidatorInput(
            title="Meeting",
            date="2027-04-01",
            start_time="14:00",
            end_time="10:00",
            duration_minutes=30,
            confidence=0.9,
        )
        result = await tool.execute(inp, tool_context)
        assert result.is_valid is True
        assert any("before start_time" in w for w in result.warnings)
        assert result.corrected_end_time == "14:30"

    async def test_invalid_end_time_format_warning(self, tool_context):
        tool = EventValidatorTool()
        inp = EventValidatorInput(
            title="Meeting",
            date="2027-04-01",
            start_time="10:00",
            end_time="not-a-time",
            duration_minutes=45,
            confidence=0.9,
        )
        result = await tool.execute(inp, tool_context)
        assert result.is_valid is True
        assert any("Invalid end_time format" in w for w in result.warnings)
        assert result.corrected_end_time == "10:45"

    async def test_low_confidence_needs_confirmation(self, tool_context):
        tool = EventValidatorTool()
        inp = EventValidatorInput(
            title="Maybe Meeting",
            date="2027-04-01",
            start_time="10:00",
            confidence=0.4,
        )
        result = await tool.execute(inp, tool_context)
        assert result.needs_confirmation is True

    async def test_high_confidence_no_confirmation(self, tool_context):
        tool = EventValidatorTool()
        inp = EventValidatorInput(
            title="Meeting",
            date="2027-04-01",
            start_time="10:00",
            confidence=0.8,
        )
        result = await tool.execute(inp, tool_context)
        assert result.needs_confirmation is False

    async def test_default_duration_used_when_zero(self, tool_context):
        tool = EventValidatorTool()
        inp = EventValidatorInput(
            title="Meeting",
            date="2027-04-01",
            start_time="10:00",
            duration_minutes=0,
            default_duration=45,
            confidence=0.9,
        )
        result = await tool.execute(inp, tool_context)
        assert result.corrected_duration_minutes == 45
        assert result.corrected_end_time == "10:45"

    async def test_duration_calculated_from_end_time(self, tool_context):
        tool = EventValidatorTool()
        inp = EventValidatorInput(
            title="Meeting",
            date="2027-04-01",
            start_time="10:00",
            end_time="11:30",
            duration_minutes=30,
            confidence=0.9,
        )
        result = await tool.execute(inp, tool_context)
        # Duration should be recalculated from end_time - start_time
        assert result.corrected_duration_minutes == 90


# ===========================================================================
# DuplicateCheckerTool
# ===========================================================================


@pytest.mark.unit
class TestDuplicateCheckerTool:
    async def test_properties(self):
        store = CalendarAgentStore()
        tool = DuplicateCheckerTool(store)
        assert tool.name == "duplicate_checker"
        assert "deduplication" in tool.tags

    async def test_not_duplicate(self, tool_context):
        store = CalendarAgentStore()
        tool = DuplicateCheckerTool(store)
        inp = DuplicateCheckerInput(
            message_id="msg-001",
            title="Meeting",
            date="2027-04-01",
            start_time="10:00",
        )
        result = await tool.execute(inp, tool_context)
        assert result.is_duplicate is False
        assert result.duplicate_reason == ""

    async def test_duplicate_by_message_id(self, tool_context):
        store = CalendarAgentStore()
        store.add_processed_email(EmailRecord(message_id="msg-001"))
        tool = DuplicateCheckerTool(store)
        inp = DuplicateCheckerInput(
            message_id="msg-001",
            title="Meeting",
            date="2027-04-01",
            start_time="10:00",
        )
        result = await tool.execute(inp, tool_context)
        assert result.is_duplicate is True
        assert result.duplicate_reason == "message_id"

    async def test_duplicate_by_event_hash(self, tool_context):
        store = CalendarAgentStore()
        event_hash = CalendarAgentStore.compute_event_hash("Meeting", "2027-04-01", "10:00")
        store.add_created_event(
            CalendarEventRef(
                google_event_id="gev-001",
                email_message_id="msg-999",
                event_hash=event_hash,
            )
        )
        tool = DuplicateCheckerTool(store)
        inp = DuplicateCheckerInput(
            message_id="msg-002",
            title="Meeting",
            date="2027-04-01",
            start_time="10:00",
        )
        result = await tool.execute(inp, tool_context)
        assert result.is_duplicate is True
        assert result.duplicate_reason == "event_hash"


# ===========================================================================
# GoogleCalendarTool
# ===========================================================================


@pytest.mark.unit
class TestGoogleCalendarTool:
    async def test_properties(self):
        tool = GoogleCalendarTool()
        assert tool.name == "google_calendar"
        assert "google" in tool.tags

    async def test_no_google_libs(self, tool_context):
        tool = GoogleCalendarTool()
        with patch(
            "ia_agent_fwk.tools.builtin.calendar_tools._has_google_libs",
            return_value=False,
        ):
            inp = GoogleCalendarInput(action="create", title="Test")
            result = await tool.execute(inp, tool_context)
        assert result.success is False
        assert "not installed" in result.error

    async def test_create_event_success(self, tool_context):
        tool = GoogleCalendarTool()
        mock_service = MagicMock()
        mock_events = MagicMock()
        mock_insert = MagicMock()
        mock_insert.execute.return_value = {
            "id": "gev-123",
            "htmlLink": "https://calendar.google.com/event?id=gev-123",
        }
        mock_events.insert.return_value = mock_insert
        mock_service.events.return_value = mock_events
        tool._service = mock_service

        with patch(
            "ia_agent_fwk.tools.builtin.calendar_tools._has_google_libs",
            return_value=True,
        ):
            inp = GoogleCalendarInput(
                action="create",
                title="Team Meeting",
                date="2027-04-01",
                start_time="10:00",
                end_time="11:00",
                timezone="America/Mexico_City",
            )
            result = await tool.execute(inp, tool_context)

        assert result.success is True
        assert result.event_id == "gev-123"
        assert "calendar.google.com" in result.event_link

    async def test_create_event_api_error(self, tool_context):
        tool = GoogleCalendarTool()
        mock_service = MagicMock()
        mock_events = MagicMock()
        mock_insert = MagicMock()
        mock_insert.execute.side_effect = RuntimeError("API quota exceeded")
        mock_events.insert.return_value = mock_insert
        mock_service.events.return_value = mock_events
        tool._service = mock_service

        with patch(
            "ia_agent_fwk.tools.builtin.calendar_tools._has_google_libs",
            return_value=True,
        ):
            inp = GoogleCalendarInput(action="create", title="Test")
            result = await tool.execute(inp, tool_context)

        assert result.success is False
        assert "API quota exceeded" in result.error

    async def test_update_event_success(self, tool_context):
        tool = GoogleCalendarTool()
        mock_service = MagicMock()
        mock_events = MagicMock()
        mock_update = MagicMock()
        mock_update.execute.return_value = {"id": "gev-123", "htmlLink": "https://link"}
        mock_events.update.return_value = mock_update
        mock_service.events.return_value = mock_events
        tool._service = mock_service

        with patch(
            "ia_agent_fwk.tools.builtin.calendar_tools._has_google_libs",
            return_value=True,
        ):
            inp = GoogleCalendarInput(
                action="update",
                event_id="gev-123",
                title="Updated Meeting",
            )
            result = await tool.execute(inp, tool_context)

        assert result.success is True

    async def test_update_without_event_id(self, tool_context):
        tool = GoogleCalendarTool()
        mock_service = MagicMock()
        tool._service = mock_service

        with patch(
            "ia_agent_fwk.tools.builtin.calendar_tools._has_google_libs",
            return_value=True,
        ):
            inp = GoogleCalendarInput(action="update", title="Test")
            result = await tool.execute(inp, tool_context)

        assert result.success is False
        assert "event_id is required" in result.error

    async def test_delete_event_success(self, tool_context):
        tool = GoogleCalendarTool()
        mock_service = MagicMock()
        mock_events = MagicMock()
        mock_delete = MagicMock()
        mock_delete.execute.return_value = None
        mock_events.delete.return_value = mock_delete
        mock_service.events.return_value = mock_events
        tool._service = mock_service

        with patch(
            "ia_agent_fwk.tools.builtin.calendar_tools._has_google_libs",
            return_value=True,
        ):
            inp = GoogleCalendarInput(action="delete", event_id="gev-123")
            result = await tool.execute(inp, tool_context)

        assert result.success is True
        assert result.event_id == "gev-123"

    async def test_delete_without_event_id(self, tool_context):
        tool = GoogleCalendarTool()
        mock_service = MagicMock()
        tool._service = mock_service

        with patch(
            "ia_agent_fwk.tools.builtin.calendar_tools._has_google_libs",
            return_value=True,
        ):
            inp = GoogleCalendarInput(action="delete")
            result = await tool.execute(inp, tool_context)

        assert result.success is False
        assert "event_id is required" in result.error

    async def test_unknown_action(self, tool_context):
        tool = GoogleCalendarTool()
        mock_service = MagicMock()
        tool._service = mock_service

        with patch(
            "ia_agent_fwk.tools.builtin.calendar_tools._has_google_libs",
            return_value=True,
        ):
            inp = GoogleCalendarInput(action="list")
            result = await tool.execute(inp, tool_context)

        assert result.success is False
        assert "Unknown action" in result.error

    async def test_service_init_error(self, tool_context):
        tool = GoogleCalendarTool()

        with (
            patch(
                "ia_agent_fwk.tools.builtin.calendar_tools._has_google_libs",
                return_value=True,
            ),
            patch.object(
                tool,
                "_get_service",
                side_effect=RuntimeError("No token.json"),
            ),
        ):
            inp = GoogleCalendarInput(action="create", title="Test")
            result = await tool.execute(inp, tool_context)

        assert result.success is False
        assert "Failed to initialize" in result.error

    def test_build_event_body_basic(self):
        inp = GoogleCalendarInput(
            action="create",
            title="Meeting",
            description="Team sync",
            date="2027-04-01",
            start_time="10:00",
            end_time="11:00",
            timezone="UTC",
        )
        body = GoogleCalendarTool._build_event_body(inp)
        assert body["summary"] == "Meeting"
        assert body["start"]["dateTime"] == "2027-04-01T10:00:00"
        assert body["end"]["dateTime"] == "2027-04-01T11:00:00"

    def test_build_event_body_with_attendees(self):
        inp = GoogleCalendarInput(
            action="create",
            title="Meeting",
            attendees=["a@b.com", "c@d.com"],
        )
        body = GoogleCalendarTool._build_event_body(inp)
        assert len(body["attendees"]) == 2
        assert body["attendees"][0] == {"email": "a@b.com"}

    def test_build_event_body_with_location(self):
        inp = GoogleCalendarInput(
            action="create",
            title="Meeting",
            location="Room A",
        )
        body = GoogleCalendarTool._build_event_body(inp)
        assert body["location"] == "Room A"

    def test_build_event_body_with_meeting_link(self):
        inp = GoogleCalendarInput(
            action="create",
            title="Meeting",
            description="Sync call",
            meeting_link="https://zoom.us/j/123",
        )
        body = GoogleCalendarTool._build_event_body(inp)
        assert "zoom.us" in body["description"]
        assert "Sync call" in body["description"]

    def test_build_event_body_no_end_time(self):
        inp = GoogleCalendarInput(
            action="create",
            title="Meeting",
            date="2027-04-01",
            start_time="10:00",
            timezone="UTC",
        )
        body = GoogleCalendarTool._build_event_body(inp)
        # When no end_time, should use start_time
        assert body["end"]["dateTime"] == "2027-04-01T10:00:00"

    async def test_get_service_caches(self, tool_context):
        tool = GoogleCalendarTool()
        sentinel = MagicMock()
        tool._service = sentinel
        with patch(
            "ia_agent_fwk.tools.builtin.calendar_tools._has_google_libs",
            return_value=True,
        ):
            svc = tool._get_service()
        assert svc is sentinel

    async def test_get_service_no_token_file(self, tool_context, tmp_path):
        import sys

        # Create mock google modules so local `from X import Y` works
        mock_request_mod = MagicMock()
        mock_request_mod.Request = MagicMock()

        mock_transport_mod = MagicMock()
        mock_transport_mod.requests = mock_request_mod

        mock_auth_mod = MagicMock()
        mock_auth_mod.transport = mock_transport_mod

        mock_creds_mod = MagicMock()
        mock_creds_mod.Credentials = MagicMock()
        mock_creds_mod.Credentials.from_authorized_user_file = MagicMock(return_value=None)

        mock_oauth2_mod = MagicMock()
        mock_oauth2_mod.credentials = mock_creds_mod

        mock_google = MagicMock()
        mock_google.auth = mock_auth_mod
        mock_google.oauth2 = mock_oauth2_mod

        mock_discovery = MagicMock()
        mock_discovery.build = MagicMock()

        mock_googleapiclient = MagicMock()
        mock_googleapiclient.discovery = mock_discovery

        modules = {
            "google": mock_google,
            "google.auth": mock_auth_mod,
            "google.auth.transport": mock_transport_mod,
            "google.auth.transport.requests": mock_request_mod,
            "google.oauth2": mock_oauth2_mod,
            "google.oauth2.credentials": mock_creds_mod,
            "googleapiclient": mock_googleapiclient,
            "googleapiclient.discovery": mock_discovery,
        }

        tool = GoogleCalendarTool(
            token_path=tmp_path / "nonexistent_token.json",
        )
        with patch.dict(sys.modules, modules), pytest.raises(Exception, match="No token.json"):
            tool._get_service()

    async def test_get_service_refreshes_expired_token(self, tool_context, tmp_path):
        import sys
        from types import ModuleType
        from unittest.mock import PropertyMock

        mock_creds = MagicMock()
        type(mock_creds).expired = PropertyMock(return_value=True)
        mock_creds.refresh_token = "refresh-tok"
        mock_creds.to_json.return_value = '{"token": "new"}'

        mock_google = MagicMock(spec=ModuleType)
        mock_oauth2 = MagicMock(spec=ModuleType)
        mock_creds_mod = MagicMock(spec=ModuleType)
        mock_creds_mod.Credentials = MagicMock()
        mock_creds_mod.Credentials.from_authorized_user_file = MagicMock(return_value=mock_creds)
        mock_oauth2.credentials = mock_creds_mod
        mock_google.oauth2 = mock_oauth2
        mock_google.auth = MagicMock(spec=ModuleType)
        mock_google.auth.transport = MagicMock(spec=ModuleType)
        mock_request = MagicMock()
        mock_google.auth.transport.requests = MagicMock(spec=ModuleType)
        mock_google.auth.transport.requests.Request = MagicMock(return_value=mock_request)

        mock_service = MagicMock()
        mock_googleapiclient = MagicMock(spec=ModuleType)
        mock_discovery = MagicMock(spec=ModuleType)
        mock_discovery.build = MagicMock(return_value=mock_service)
        mock_googleapiclient.discovery = mock_discovery

        modules = {
            "google": mock_google,
            "google.auth": mock_google.auth,
            "google.auth.transport": mock_google.auth.transport,
            "google.auth.transport.requests": mock_google.auth.transport.requests,
            "google.oauth2": mock_oauth2,
            "google.oauth2.credentials": mock_creds_mod,
            "googleapiclient": mock_googleapiclient,
            "googleapiclient.discovery": mock_discovery,
        }

        tool = GoogleCalendarTool(token_path=tmp_path / "token.json")
        (tmp_path / "token.json").write_text('{"token": "old"}')

        with patch.dict(sys.modules, modules):
            svc = tool._get_service()

        mock_creds.refresh.assert_called_once()
        assert svc is mock_service


# ===========================================================================
# F-001: EventExtractor graceful fallback on non-numeric values
# ===========================================================================


@pytest.mark.unit
class TestEventExtractorEdgeCases:
    async def test_non_numeric_duration_falls_back_to_default(self, tool_context):
        response_json = json.dumps(
            {
                "detection_status": "EVENT_FOUND",
                "title": "Meeting",
                "duration_minutes": "about thirty",
                "confidence": 0.8,
            }
        )
        provider = _make_mock_provider(response_json)
        tool = EventExtractorTool(provider)
        inp = EventExtractorInput(clean_email_text="Meeting at 10am")
        result = await tool.execute(inp, tool_context)
        assert result.duration_minutes == 30  # Fallback default
        assert result.detection_status == "EVENT_FOUND"

    async def test_non_numeric_confidence_falls_back_to_zero(self, tool_context):
        response_json = json.dumps(
            {
                "detection_status": "EVENT_FOUND",
                "title": "Meeting",
                "confidence": "high",
            }
        )
        provider = _make_mock_provider(response_json)
        tool = EventExtractorTool(provider)
        inp = EventExtractorInput(clean_email_text="Meeting")
        result = await tool.execute(inp, tool_context)
        assert result.confidence == 0.0  # Fallback default
