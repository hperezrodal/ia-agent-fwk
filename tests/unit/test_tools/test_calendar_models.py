"""Tests for calendar agent data models and CalendarAgentStore."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ia_agent_fwk.tools.builtin.calendar_models import (
    CalendarAgentStore,
    CalendarEventRef,
    EmailRecord,
    ExtractedEvent,
    PendingConfirmation,
    UserCorrection,
)


@pytest.mark.unit
class TestExtractedEvent:
    def test_defaults(self):
        event = ExtractedEvent()
        assert event.title == ""
        assert event.date == ""
        assert event.start_time == ""
        assert event.end_time == ""
        assert event.duration_minutes == 30
        assert event.location == ""
        assert event.meeting_link == ""
        assert event.participants == []
        assert event.description == ""
        assert event.confidence == 0.0
        assert event.detection_status == "NO_EVENT"

    def test_custom_values(self):
        event = ExtractedEvent(
            title="Standup",
            date="2026-04-01",
            start_time="09:00",
            end_time="09:30",
            duration_minutes=30,
            location="Zoom",
            participants=["a@b.com"],
            confidence=0.95,
            detection_status="EVENT_FOUND",
        )
        assert event.title == "Standup"
        assert event.participants == ["a@b.com"]
        assert event.confidence == 0.95

    def test_frozen(self):
        event = ExtractedEvent(title="Test")
        with pytest.raises(Exception):  # noqa: B017
            event.title = "Changed"  # type: ignore[misc]


@pytest.mark.unit
class TestEmailRecord:
    def test_required_message_id(self):
        record = EmailRecord(message_id="msg-001")
        assert record.message_id == "msg-001"
        assert record.status == "processed"
        assert record.event_hash is None

    def test_full_record(self):
        record = EmailRecord(
            message_id="msg-002",
            subject="Meeting Invite",
            sender="alice@test.com",
            received_at="2026-03-11T10:00:00Z",
            status="error",
            event_hash="abc123",
        )
        assert record.subject == "Meeting Invite"
        assert record.event_hash == "abc123"


@pytest.mark.unit
class TestCalendarEventRef:
    def test_creation(self):
        ref = CalendarEventRef(
            google_event_id="gev-001",
            email_message_id="msg-001",
            event_hash="hash-001",
            title="Test Event",
        )
        assert ref.google_event_id == "gev-001"
        assert ref.created_at == ""


@pytest.mark.unit
class TestUserCorrection:
    def test_creation(self):
        correction = UserCorrection(
            email_snippet="Meeting at 3pm",
            original_extraction={"start_time": "03:00"},
            corrected_fields={"start_time": "15:00"},
            created_at="2026-03-11T10:00:00Z",
        )
        assert correction.email_snippet == "Meeting at 3pm"
        assert correction.corrected_fields["start_time"] == "15:00"


@pytest.mark.unit
class TestPendingConfirmation:
    def test_creation(self):
        pending = PendingConfirmation(
            confirmation_id="conf-001",
            email_message_id="msg-001",
            extracted_event={"title": "Standup"},
            sent_at="2026-03-11T10:00:00Z",
        )
        assert pending.status == "pending"
        assert pending.extracted_event["title"] == "Standup"


@pytest.mark.unit
class TestCalendarAgentStore:
    def test_empty_store(self):
        store = CalendarAgentStore()
        assert store.processed_emails == {}
        assert store.created_events == {}
        assert store.corrections == []
        assert store.pending_confirmations == {}

    # -- compute_event_hash ------------------------------------------------

    def test_compute_event_hash_deterministic(self):
        h1 = CalendarAgentStore.compute_event_hash("Meeting", "2026-04-01", "10:00")
        h2 = CalendarAgentStore.compute_event_hash("Meeting", "2026-04-01", "10:00")
        assert h1 == h2
        assert len(h1) == 16

    def test_compute_event_hash_case_insensitive_title(self):
        h1 = CalendarAgentStore.compute_event_hash("Meeting", "2026-04-01", "10:00")
        h2 = CalendarAgentStore.compute_event_hash("meeting", "2026-04-01", "10:00")
        assert h1 == h2

    def test_compute_event_hash_strips_whitespace(self):
        h1 = CalendarAgentStore.compute_event_hash("Meeting", "2026-04-01", "10:00")
        h2 = CalendarAgentStore.compute_event_hash("  Meeting  ", "2026-04-01", "10:00")
        assert h1 == h2

    def test_compute_event_hash_different_inputs(self):
        h1 = CalendarAgentStore.compute_event_hash("Meeting A", "2026-04-01", "10:00")
        h2 = CalendarAgentStore.compute_event_hash("Meeting B", "2026-04-01", "10:00")
        assert h1 != h2

    # -- is_duplicate ------------------------------------------------------

    def test_is_duplicate_by_message_id(self):
        store = CalendarAgentStore()
        store.add_processed_email(EmailRecord(message_id="msg-001"))
        assert store.is_duplicate("msg-001", "any-hash") is True

    def test_is_duplicate_by_event_hash(self):
        store = CalendarAgentStore()
        store.add_created_event(
            CalendarEventRef(
                google_event_id="gev-001",
                email_message_id="msg-001",
                event_hash="hash-001",
            )
        )
        assert store.is_duplicate("msg-999", "hash-001") is True

    def test_not_duplicate(self):
        store = CalendarAgentStore()
        assert store.is_duplicate("msg-001", "hash-001") is False

    def test_is_duplicate_empty_message_id(self):
        store = CalendarAgentStore()
        store.add_created_event(
            CalendarEventRef(
                google_event_id="gev-001",
                email_message_id="msg-001",
                event_hash="hash-001",
            )
        )
        # Empty message_id should not match — fall through to event_hash
        assert store.is_duplicate("", "hash-001") is True
        assert store.is_duplicate("", "hash-999") is False

    # -- add_processed_email -----------------------------------------------

    def test_add_processed_email(self):
        store = CalendarAgentStore()
        record = EmailRecord(message_id="msg-001", subject="Test")
        store.add_processed_email(record)
        assert "msg-001" in store.processed_emails
        assert store.processed_emails["msg-001"].subject == "Test"

    def test_add_processed_email_overwrites(self):
        store = CalendarAgentStore()
        store.add_processed_email(EmailRecord(message_id="msg-001", status="processed"))
        store.add_processed_email(EmailRecord(message_id="msg-001", status="error"))
        assert store.processed_emails["msg-001"].status == "error"

    # -- add_created_event -------------------------------------------------

    def test_add_created_event(self):
        store = CalendarAgentStore()
        ref = CalendarEventRef(
            google_event_id="gev-001",
            email_message_id="msg-001",
            event_hash="hash-001",
        )
        store.add_created_event(ref)
        assert "hash-001" in store.created_events

    # -- corrections -------------------------------------------------------

    def test_add_correction(self):
        store = CalendarAgentStore()
        correction = UserCorrection(
            email_snippet="test",
            original_extraction={"a": 1},
            corrected_fields={"a": 2},
        )
        store.add_correction(correction)
        assert len(store.corrections) == 1

    def test_add_correction_caps_at_max(self):
        store = CalendarAgentStore()
        for i in range(25):
            store.add_correction(
                UserCorrection(email_snippet=f"snippet-{i}"),
                max_corrections=20,
            )
        assert len(store.corrections) == 20
        # Should keep the most recent
        assert store.corrections[0].email_snippet == "snippet-5"
        assert store.corrections[-1].email_snippet == "snippet-24"

    def test_get_relevant_corrections_limit(self):
        store = CalendarAgentStore()
        for i in range(10):
            store.add_correction(UserCorrection(email_snippet=f"snippet-{i}"))
        recent = store.get_relevant_corrections(limit=3)
        assert len(recent) == 3
        assert recent[0].email_snippet == "snippet-7"
        assert recent[-1].email_snippet == "snippet-9"

    def test_get_relevant_corrections_fewer_than_limit(self):
        store = CalendarAgentStore()
        store.add_correction(UserCorrection(email_snippet="only"))
        recent = store.get_relevant_corrections(limit=5)
        assert len(recent) == 1

    # -- pending confirmations ---------------------------------------------

    def test_add_pending_confirmation(self):
        store = CalendarAgentStore()
        pending = PendingConfirmation(
            confirmation_id="conf-001",
            email_message_id="msg-001",
        )
        store.add_pending_confirmation(pending)
        assert "conf-001" in store.pending_confirmations

    def test_resolve_confirmation(self):
        store = CalendarAgentStore()
        pending = PendingConfirmation(
            confirmation_id="conf-001",
            email_message_id="msg-001",
        )
        store.add_pending_confirmation(pending)
        store.resolve_confirmation("conf-001", "confirmed")
        assert store.pending_confirmations["conf-001"].status == "confirmed"

    def test_resolve_confirmation_preserves_fields(self):
        store = CalendarAgentStore()
        pending = PendingConfirmation(
            confirmation_id="conf-001",
            email_message_id="msg-001",
            extracted_event={"title": "Standup"},
            sent_at="2026-03-11T10:00:00Z",
        )
        store.add_pending_confirmation(pending)
        store.resolve_confirmation("conf-001", "cancelled")
        resolved = store.pending_confirmations["conf-001"]
        assert resolved.email_message_id == "msg-001"
        assert resolved.extracted_event == {"title": "Standup"}
        assert resolved.sent_at == "2026-03-11T10:00:00Z"

    def test_resolve_confirmation_unknown_id(self):
        store = CalendarAgentStore()
        # Should not raise
        store.resolve_confirmation("nonexistent", "confirmed")
        assert "nonexistent" not in store.pending_confirmations

    # -- persistence -------------------------------------------------------

    def test_save_and_load(self, tmp_path: Path):
        store = CalendarAgentStore()
        store.add_processed_email(EmailRecord(message_id="msg-001", subject="Meeting"))
        store.add_created_event(
            CalendarEventRef(
                google_event_id="gev-001",
                email_message_id="msg-001",
                event_hash="hash-001",
                title="Meeting",
            )
        )
        store.add_correction(
            UserCorrection(
                email_snippet="test",
                original_extraction={"a": 1},
                corrected_fields={"a": 2},
            )
        )
        store.add_pending_confirmation(
            PendingConfirmation(
                confirmation_id="conf-001",
                email_message_id="msg-001",
            )
        )

        path = tmp_path / "store.json"
        store.save(path)
        assert path.exists()

        # Verify JSON is valid
        data = json.loads(path.read_text())
        assert "processed_emails" in data
        assert "created_events" in data
        assert "corrections" in data
        assert "pending_confirmations" in data

        # Load into a new store
        store2 = CalendarAgentStore()
        store2.load(path)
        assert len(store2.processed_emails) == 1
        assert store2.processed_emails["msg-001"].subject == "Meeting"
        assert len(store2.created_events) == 1
        assert store2.created_events["hash-001"].title == "Meeting"
        assert len(store2.corrections) == 1
        assert len(store2.pending_confirmations) == 1

    def test_load_missing_file(self, tmp_path: Path):
        store = CalendarAgentStore()
        store.load(tmp_path / "nonexistent.json")
        # Should be no-op
        assert store.processed_emails == {}

    def test_load_corrupt_json(self, tmp_path: Path):
        path = tmp_path / "corrupt.json"
        path.write_text("not valid json {{{")
        store = CalendarAgentStore()
        store.load(path)
        # Should be no-op (warns but doesn't crash)
        assert store.processed_emails == {}

    def test_save_creates_parent_dirs(self, tmp_path: Path):
        store = CalendarAgentStore()
        path = tmp_path / "sub" / "dir" / "store.json"
        store.save(path)
        assert path.exists()

    def test_roundtrip_empty_store(self, tmp_path: Path):
        store = CalendarAgentStore()
        path = tmp_path / "empty.json"
        store.save(path)
        store2 = CalendarAgentStore()
        store2.load(path)
        assert store2.processed_emails == {}
        assert store2.created_events == {}
        assert store2.corrections == []
        assert store2.pending_confirmations == {}
