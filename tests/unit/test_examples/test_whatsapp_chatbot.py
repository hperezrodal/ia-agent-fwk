"""Unit tests for the WhatsApp auto insurance sales chatbot example."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from examples.whatsapp_chatbot.agent import (
    WhatsAppSalesBotAgent,
    create_whatsapp_sales_agent,
)
from examples.whatsapp_chatbot.models import (
    InsuranceQuote,
    SalesConversationState,
    VehicleData,
    is_valid_plate,
    normalize_plate,
)
from examples.whatsapp_chatbot.services import MockQuotingService
from examples.whatsapp_chatbot.tools import (
    HumanHandoffInput,
    HumanHandoffTool,
    QuoteRequestInput,
    QuoteRequestTool,
    SalesFAQSearchInput,
    SalesFAQSearchTool,
    VehicleLookupInput,
    VehicleLookupTool,
)
from examples.whatsapp_chatbot.whatsapp_adapter import (
    WhatsAppInteractiveAdapter,
)

from ia_agent_fwk.integrations.whatsapp import WhatsAppIntegration
from ia_agent_fwk.tools.base import ToolContext
from ia_agent_fwk.tools.exceptions import ToolExecutionError

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "examples" / "whatsapp_chatbot" / "data"


@pytest.fixture
def tool_context() -> ToolContext:
    return ToolContext(execution_id="test-exec", agent_id="test-agent", timeout=30)


@pytest.fixture
def mock_quoting_service() -> MockQuotingService:
    return MockQuotingService(data_dir=_DATA_DIR)


@pytest.fixture
def mock_provider() -> MagicMock:
    provider = MagicMock()
    provider.provider_name = "ollama"
    provider.chat = AsyncMock(return_value=MagicMock(content="Hola!"))
    return provider


@pytest.fixture
def whatsapp_integration() -> WhatsAppIntegration:
    return WhatsAppIntegration(
        access_token="test-token",
        phone_number_id="123456789",
        verify_token="test-verify",
    )


# ---------------------------------------------------------------------------
# TestModels
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestModels:
    """Tests for domain models."""

    def test_valid_new_plate(self) -> None:
        assert is_valid_plate("AB123CD")

    def test_valid_old_plate(self) -> None:
        assert is_valid_plate("ABC123")

    def test_valid_plate_with_spaces(self) -> None:
        assert is_valid_plate("AB 123 CD")

    def test_valid_plate_lowercase(self) -> None:
        assert is_valid_plate("ab123cd")

    def test_invalid_plate(self) -> None:
        assert not is_valid_plate("INVALID")
        assert not is_valid_plate("12345")
        assert not is_valid_plate("")
        assert not is_valid_plate("A")

    def test_normalize_plate(self) -> None:
        assert normalize_plate("ab 123 cd") == "AB123CD"
        assert normalize_plate("AB-123-CD") == "AB123CD"

    def test_vehicle_data_plate_normalization(self) -> None:
        v = VehicleData(plate="ab 123 cd")
        assert v.plate == "AB123CD"

    def test_vehicle_data_complete_for_quote(self) -> None:
        v = VehicleData(brand="VW", model="Gol", year=2020, use="particular")
        assert v.is_complete_for_quote()

    def test_vehicle_data_incomplete_for_quote(self) -> None:
        v = VehicleData(brand="VW")
        assert not v.is_complete_for_quote()

    def test_insurance_quote_creation(self) -> None:
        q = InsuranceQuote(
            quote_id="Q-001",
            insurer_name="Test",
            coverage_type="TC",
            monthly_premium=15000,
            annual_premium=170000,
            deductible=50000,
        )
        assert q.quote_id == "Q-001"
        assert q.key_benefits == []

    def test_sales_state_serialization(self) -> None:
        state = SalesConversationState(
            workflow_step="quoting",
            vehicle_data=VehicleData(plate="AB123CD", brand="VW", model="Gol", year=2020, use="particular"),
        )
        d = state.to_dict()
        restored = SalesConversationState.from_dict(d)
        assert restored.workflow_step == "quoting"
        assert restored.vehicle_data is not None
        assert restored.vehicle_data.brand == "VW"

    def test_sales_state_context_summary(self) -> None:
        state = SalesConversationState(
            workflow_step="quote_presentation",
            vehicle_data=VehicleData(brand="Fiat", model="Cronos", year=2022, use="particular"),
            quotes=[
                InsuranceQuote(
                    quote_id="Q-001",
                    insurer_name="Test Insurer",
                    coverage_type="TC",
                    monthly_premium=15000,
                    annual_premium=170000,
                    deductible=50000,
                ),
            ],
        )
        summary = state.to_context_summary()
        assert "quote_presentation" in summary
        assert "Fiat" in summary
        assert "Test Insurer" in summary
        assert "15,000" in summary


# ---------------------------------------------------------------------------
# TestMockQuotingService
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMockQuotingService:
    """Tests for the mock quoting service."""

    async def test_lookup_vehicle_found(self, mock_quoting_service: MockQuotingService) -> None:
        result = await mock_quoting_service.lookup_vehicle("AB123CD")
        assert result is not None
        assert result.brand == "Volkswagen"
        assert result.model == "Gol Trend"
        assert result.year == 2020

    async def test_lookup_vehicle_not_found(self, mock_quoting_service: MockQuotingService) -> None:
        result = await mock_quoting_service.lookup_vehicle("ZZ999ZZ")
        assert result is None

    async def test_lookup_vehicle_old_format(self, mock_quoting_service: MockQuotingService) -> None:
        result = await mock_quoting_service.lookup_vehicle("ABC123")
        assert result is not None
        assert result.brand == "Fiat"
        assert result.model == "Palio"

    async def test_get_quotes_particular(self, mock_quoting_service: MockQuotingService) -> None:
        vehicle = VehicleData(brand="VW", model="Gol", year=2020, use="particular")
        quotes = await mock_quoting_service.get_quotes(vehicle)
        assert len(quotes) > 0
        assert all(q.monthly_premium > 0 for q in quotes)

    async def test_get_quotes_comercial(self, mock_quoting_service: MockQuotingService) -> None:
        vehicle = VehicleData(brand="Toyota", model="Hilux", year=2021, use="comercial")
        quotes = await mock_quoting_service.get_quotes(vehicle)
        assert len(quotes) > 0

    async def test_get_quotes_no_data_dir(self) -> None:
        svc = MockQuotingService(data_dir=Path("/nonexistent"))
        vehicle = VehicleData(brand="VW", model="Gol", year=2020, use="particular")
        quotes = await svc.get_quotes(vehicle)
        assert quotes == []


# ---------------------------------------------------------------------------
# TestVehicleLookupTool
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestVehicleLookupTool:
    """Tests for the vehicle lookup tool."""

    async def test_valid_plate_found(self, mock_quoting_service: MockQuotingService, tool_context: ToolContext) -> None:
        tool = VehicleLookupTool(mock_quoting_service)
        result = await tool.execute(VehicleLookupInput(plate="AB123CD"), tool_context)
        assert result.found is True
        assert result.brand == "Volkswagen"

    async def test_valid_plate_not_found(
        self, mock_quoting_service: MockQuotingService, tool_context: ToolContext
    ) -> None:
        tool = VehicleLookupTool(mock_quoting_service)
        result = await tool.execute(VehicleLookupInput(plate="ZZ999ZZ"), tool_context)
        assert result.found is False
        assert "No se encontro" in result.error

    async def test_invalid_plate(self, mock_quoting_service: MockQuotingService, tool_context: ToolContext) -> None:
        tool = VehicleLookupTool(mock_quoting_service)
        result = await tool.execute(VehicleLookupInput(plate="INVALID"), tool_context)
        assert result.found is False
        assert "invalido" in result.error
        # F-002: plate value should NOT appear in error message (PII masking)
        assert "INVALID" not in result.error

    async def test_tool_properties(self, mock_quoting_service: MockQuotingService) -> None:
        tool = VehicleLookupTool(mock_quoting_service)
        assert tool.name == "vehicle_lookup"
        assert tool.input_schema is VehicleLookupInput
        assert tool.output_schema is not None
        assert "sales" in tool.tags


# ---------------------------------------------------------------------------
# TestQuoteRequestTool
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestQuoteRequestTool:
    """Tests for the quote request tool."""

    async def test_quotes_found(self, mock_quoting_service: MockQuotingService, tool_context: ToolContext) -> None:
        tool = QuoteRequestTool(mock_quoting_service)
        result = await tool.execute(
            QuoteRequestInput(brand="Volkswagen", model="Gol Trend", year=2020, use="particular"),
            tool_context,
        )
        assert result.total_found > 0
        assert len(result.quotes) > 0
        assert all(q.monthly_premium > 0 for q in result.quotes)

    async def test_no_quotes(self, tool_context: ToolContext) -> None:
        svc = MockQuotingService(data_dir=Path("/nonexistent"))
        tool = QuoteRequestTool(svc)
        result = await tool.execute(
            QuoteRequestInput(brand="Unknown", model="X", year=2020, use="particular"),
            tool_context,
        )
        assert result.total_found == 0
        assert "No se encontraron" in result.error

    async def test_tool_properties(self, mock_quoting_service: MockQuotingService) -> None:
        tool = QuoteRequestTool(mock_quoting_service)
        assert tool.name == "quote_request"
        assert "sales" in tool.tags


# ---------------------------------------------------------------------------
# TestFAQSearchTool
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSalesFAQSearchTool:
    """Tests for the FAQ search tool."""

    async def test_matching_query(self, tool_context: ToolContext) -> None:
        tool = SalesFAQSearchTool.from_file(_DATA_DIR / "faq.json")
        result = await tool.execute(
            SalesFAQSearchInput(query="responsabilidad civil obligatoria"),
            tool_context,
        )
        assert result.total_found > 0
        assert result.results[0].relevance_score > 0

    async def test_no_results(self, tool_context: ToolContext) -> None:
        tool = SalesFAQSearchTool.from_file(_DATA_DIR / "faq.json")
        result = await tool.execute(
            SalesFAQSearchInput(query="xyznonexistent"),
            tool_context,
        )
        assert result.total_found == 0

    async def test_max_results(self, tool_context: ToolContext) -> None:
        tool = SalesFAQSearchTool.from_file(_DATA_DIR / "faq.json")
        result = await tool.execute(
            SalesFAQSearchInput(query="seguro auto cobertura", max_results=2),
            tool_context,
        )
        assert len(result.results) <= 2

    async def test_empty_faq(self, tool_context: ToolContext) -> None:
        tool = SalesFAQSearchTool()
        result = await tool.execute(
            SalesFAQSearchInput(query="anything"),
            tool_context,
        )
        assert result.total_found == 0

    def test_tool_properties(self) -> None:
        tool = SalesFAQSearchTool()
        assert tool.name == "faq_search"
        assert "faq" in tool.tags


# ---------------------------------------------------------------------------
# TestHumanHandoffTool
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestHumanHandoffTool:
    """Tests for the human handoff tool."""

    async def test_successful_handoff(self, tool_context: ToolContext) -> None:
        tool = HumanHandoffTool()
        result = await tool.execute(
            HumanHandoffInput(
                reason="Cliente quiere contratar",
                conversation_summary="Cotizo seguro VW Gol 2020",
            ),
            tool_context,
        )
        assert result.status == "transferred"
        assert result.handoff_id == "HO-0001"
        assert "transferida" in result.message

    async def test_counter_increments(self, tool_context: ToolContext) -> None:
        tool = HumanHandoffTool()
        r1 = await tool.execute(
            HumanHandoffInput(reason="r1", conversation_summary="s1"),
            tool_context,
        )
        r2 = await tool.execute(
            HumanHandoffInput(reason="r2", conversation_summary="s2"),
            tool_context,
        )
        assert r1.handoff_id == "HO-0001"
        assert r2.handoff_id == "HO-0002"

    def test_tool_properties(self) -> None:
        tool = HumanHandoffTool()
        assert tool.name == "human_handoff"
        assert "handoff" in tool.tags


# ---------------------------------------------------------------------------
# TestWhatsAppInteractiveAdapter
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestWhatsAppInteractiveAdapter:
    """Tests for the WhatsApp interactive message adapter."""

    def test_button_reply_parsing(self, whatsapp_integration: WhatsAppIntegration) -> None:
        adapter = WhatsAppInteractiveAdapter(whatsapp_integration)
        event: dict[str, Any] = {
            "entry": [
                {
                    "changes": [
                        {
                            "value": {
                                "messages": [
                                    {
                                        "type": "interactive",
                                        "from": "5491155551234",
                                        "id": "msg1",
                                        "timestamp": "123",
                                        "interactive": {
                                            "type": "button_reply",
                                            "button_reply": {"id": "btn_yes", "title": "Si"},
                                        },
                                    }
                                ],
                                "metadata": {"phone_number_id": "123"},
                            }
                        }
                    ]
                }
            ],
        }
        reply_id, reply_title = adapter.process_interactive_reply(event)
        assert reply_id == "btn_yes"
        assert reply_title == "Si"

    def test_list_reply_parsing(self, whatsapp_integration: WhatsAppIntegration) -> None:
        adapter = WhatsAppInteractiveAdapter(whatsapp_integration)
        event: dict[str, Any] = {
            "entry": [
                {
                    "changes": [
                        {
                            "value": {
                                "messages": [
                                    {
                                        "type": "interactive",
                                        "from": "5491155551234",
                                        "id": "msg2",
                                        "timestamp": "456",
                                        "interactive": {
                                            "type": "list_reply",
                                            "list_reply": {"id": "Q-001", "title": "Fed. Patronal"},
                                        },
                                    }
                                ],
                                "metadata": {"phone_number_id": "123"},
                            }
                        }
                    ]
                }
            ],
        }
        reply_id, reply_title = adapter.process_interactive_reply(event)
        assert reply_id == "Q-001"
        assert reply_title == "Fed. Patronal"

    def test_text_message_not_interactive(self, whatsapp_integration: WhatsAppIntegration) -> None:
        adapter = WhatsAppInteractiveAdapter(whatsapp_integration)
        event: dict[str, Any] = {
            "entry": [
                {
                    "changes": [
                        {
                            "value": {
                                "messages": [{"type": "text", "from": "123", "text": {"body": "hi"}}],
                            }
                        }
                    ]
                }
            ],
        }
        reply_id, reply_title = adapter.process_interactive_reply(event)
        assert reply_id is None
        assert reply_title is None

    def test_empty_event(self, whatsapp_integration: WhatsAppIntegration) -> None:
        adapter = WhatsAppInteractiveAdapter(whatsapp_integration)
        reply_id, _ = adapter.process_interactive_reply({})
        assert reply_id is None

    def test_extended_incoming_interactive(self, whatsapp_integration: WhatsAppIntegration) -> None:
        adapter = WhatsAppInteractiveAdapter(whatsapp_integration)
        event: dict[str, Any] = {
            "entry": [
                {
                    "changes": [
                        {
                            "value": {
                                "messages": [
                                    {
                                        "type": "interactive",
                                        "from": "5491155551234",
                                        "id": "msg1",
                                        "timestamp": "123",
                                        "interactive": {
                                            "type": "button_reply",
                                            "button_reply": {"id": "btn_confirm", "title": "Confirmar"},
                                        },
                                    }
                                ],
                                "metadata": {"phone_number_id": "123"},
                            }
                        }
                    ]
                }
            ],
        }
        msg = adapter.process_incoming_extended(event)
        assert msg is not None
        assert msg.content == "Confirmar"
        assert msg.sender == "5491155551234"
        assert msg.metadata["interactive_reply_id"] == "btn_confirm"

    def test_extended_incoming_text_returns_none(self, whatsapp_integration: WhatsAppIntegration) -> None:
        adapter = WhatsAppInteractiveAdapter(whatsapp_integration)
        event: dict[str, Any] = {
            "entry": [
                {
                    "changes": [
                        {
                            "value": {
                                "messages": [{"type": "text", "from": "123", "text": {"body": "hi"}}],
                            }
                        }
                    ]
                }
            ],
        }
        msg = adapter.process_incoming_extended(event)
        assert msg is None

    def test_payload_construction_button(self, whatsapp_integration: WhatsAppIntegration) -> None:
        adapter = WhatsAppInteractiveAdapter(whatsapp_integration)
        payload = adapter._build_interactive_payload(
            recipient="123",
            interactive_type="button",
            body="Confirmas los datos?",
            header="Confirmacion",
            footer="",
            action={
                "buttons": [
                    {"type": "reply", "reply": {"id": "yes", "title": "Si"}},
                    {"type": "reply", "reply": {"id": "no", "title": "No"}},
                ],
            },
        )
        assert payload["type"] == "interactive"
        assert payload["messaging_product"] == "whatsapp"
        assert payload["interactive"]["type"] == "button"
        assert payload["interactive"]["header"]["text"] == "Confirmacion"
        assert len(payload["interactive"]["action"]["buttons"]) == 2

    def test_payload_construction_list(self, whatsapp_integration: WhatsAppIntegration) -> None:
        adapter = WhatsAppInteractiveAdapter(whatsapp_integration)
        payload = adapter._build_interactive_payload(
            recipient="123",
            interactive_type="list",
            body="Estas son tus cotizaciones:",
            header="",
            footer="Precios en ARS",
            action={
                "button": "Ver opciones",
                "sections": [{"title": "Cotizaciones", "rows": []}],
            },
        )
        assert payload["interactive"]["type"] == "list"
        assert payload["interactive"]["footer"]["text"] == "Precios en ARS"
        assert "header" not in payload["interactive"]


# ---------------------------------------------------------------------------
# TestWhatsAppSalesBotFactory
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestWhatsAppSalesBotFactory:
    """Tests for the agent factory function."""

    def test_create_default_agent(self, mock_provider: MagicMock) -> None:
        agent = create_whatsapp_sales_agent(provider=mock_provider)
        assert isinstance(agent, WhatsAppSalesBotAgent)
        assert agent.agent_type == "whatsapp_sales"
        assert agent._config.name == "whatsapp-sales-agent"

    def test_tools_registered(self, mock_provider: MagicMock) -> None:
        # Default conversation_state_summary puts the agent in "greeting" phase,
        # which has no tools (phase-gated to avoid small models forcing tool calls).
        agent = create_whatsapp_sales_agent(provider=mock_provider)
        assert agent._config.tools == []

    def test_system_prompt_contains_workflow(self, mock_provider: MagicMock) -> None:
        agent = create_whatsapp_sales_agent(provider=mock_provider)
        assert "Sos un asistente de ventas" in agent._config.system_prompt
        assert "greeting" in agent._config.system_prompt

    def test_custom_state_in_prompt(self, mock_provider: MagicMock) -> None:
        agent = create_whatsapp_sales_agent(
            provider=mock_provider,
            conversation_state_summary="Estado actual del flujo: quoting\nVehiculo: VW Gol 2020",
        )
        assert "quoting" in agent._config.system_prompt
        assert "VW Gol 2020" in agent._config.system_prompt

    def test_config_overrides(self, mock_provider: MagicMock) -> None:
        agent = create_whatsapp_sales_agent(
            provider=mock_provider,
            config_overrides={"max_iterations": 5, "execution_timeout": 60},
        )
        assert agent._config.max_iterations == 5
        assert agent._config.execution_timeout == 60

    def test_custom_quoting_service(self, mock_provider: MagicMock) -> None:
        svc = MockQuotingService(data_dir=_DATA_DIR)
        agent = create_whatsapp_sales_agent(
            provider=mock_provider,
            quoting_service=svc,
        )
        assert agent.agent_type == "whatsapp_sales"

    def test_custom_data_dir(self, mock_provider: MagicMock) -> None:
        agent = create_whatsapp_sales_agent(
            provider=mock_provider,
            data_dir=_DATA_DIR,
        )
        assert agent.agent_type == "whatsapp_sales"


# ---------------------------------------------------------------------------
# TestToolApiErrors (F-015)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestToolApiErrors:
    """Tests for tool error paths when QuotingService raises exceptions."""

    async def test_vehicle_lookup_api_error(self, tool_context: ToolContext) -> None:
        svc = AsyncMock()
        svc.lookup_vehicle = AsyncMock(side_effect=RuntimeError("API timeout"))
        tool = VehicleLookupTool(svc)
        with pytest.raises(ToolExecutionError, match="Error al buscar vehiculo"):
            await tool.execute(VehicleLookupInput(plate="AB123CD"), tool_context)

    async def test_quote_request_api_error(self, tool_context: ToolContext) -> None:
        svc = AsyncMock()
        svc.get_quotes = AsyncMock(side_effect=RuntimeError("Connection refused"))
        tool = QuoteRequestTool(svc)
        with pytest.raises(ToolExecutionError, match="Error al solicitar cotizaciones"):
            await tool.execute(
                QuoteRequestInput(brand="VW", model="Gol", year=2020, use="particular"),
                tool_context,
            )


# ---------------------------------------------------------------------------
# TestPgConversationBackend (F-014)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPgConversationBackend:
    """Tests for PostgreSQL conversation persistence with mocked asyncpg."""

    def _make_backend(self) -> Any:
        from examples.whatsapp_chatbot.persistence import PgConversationBackend

        backend = PgConversationBackend(
            database_url="postgresql://test:test@localhost:5432/test",
        )
        backend._table_ready = True  # Skip DDL
        return backend

    async def test_close_resets_table_ready(self) -> None:
        backend = self._make_backend()
        backend._pool = AsyncMock()
        backend._pool.close = AsyncMock()
        await backend.close()
        assert backend._pool is None
        assert backend._table_ready is False

    async def test_health_check_no_pool(self) -> None:
        from examples.whatsapp_chatbot.persistence import PgConversationBackend

        backend = PgConversationBackend(
            database_url="postgresql://bad:bad@localhost:1/bad",
        )
        # health_check should return False when pool can't connect
        result = await backend.health_check()
        assert result is False

    async def test_create_conversation_calls_execute(self) -> None:
        backend = self._make_backend()
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock()
        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)
        mock_pool = MagicMock()
        mock_pool.acquire.return_value = mock_ctx
        backend._pool = mock_pool

        conv_id = await backend.create_conversation("+5491155551234", title="Test")
        assert conv_id  # UUID string returned
        mock_conn.execute.assert_called_once()

    async def test_update_state_calls_execute(self) -> None:
        backend = self._make_backend()
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock()
        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)
        mock_pool = MagicMock()
        mock_pool.acquire.return_value = mock_ctx
        backend._pool = mock_pool

        await backend.update_state("00000000-0000-0000-0000-000000000001", {"step": "quoting"})
        mock_conn.execute.assert_called_once()


# ---------------------------------------------------------------------------
# TestQuoteRequestUseValidation (F-016)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestQuoteRequestUseValidation:
    """Tests for use field normalization in QuoteRequestInput."""

    def test_particular_passthrough(self) -> None:
        inp = QuoteRequestInput(brand="VW", model="Gol", year=2020, use="particular")
        assert inp.use == "particular"

    def test_comercial_passthrough(self) -> None:
        inp = QuoteRequestInput(brand="VW", model="Gol", year=2020, use="comercial")
        assert inp.use == "comercial"

    def test_personal_to_particular(self) -> None:
        inp = QuoteRequestInput(brand="VW", model="Gol", year=2020, use="personal")
        assert inp.use == "particular"

    def test_commercial_to_comercial(self) -> None:
        inp = QuoteRequestInput(brand="VW", model="Gol", year=2020, use="commercial")
        assert inp.use == "comercial"
