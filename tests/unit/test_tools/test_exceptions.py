"""Tests for the tool exception hierarchy."""

from ia_agent_fwk.tools.exceptions import (
    ToolError,
    ToolExecutionError,
    ToolNotFoundError,
    ToolPermissionError,
    ToolTimeoutError,
    ToolValidationError,
)


class TestToolError:
    def test_is_exception(self):
        assert issubclass(ToolError, Exception)

    def test_instantiation(self):
        err = ToolError("test error")
        assert str(err) == "test error"


class TestToolNotFoundError:
    def test_inherits_from_tool_error(self):
        assert issubclass(ToolNotFoundError, ToolError)

    def test_instantiation(self):
        err = ToolNotFoundError("tool 'foo' not found")
        assert str(err) == "tool 'foo' not found"


class TestToolValidationError:
    def test_inherits_from_tool_error(self):
        assert issubclass(ToolValidationError, ToolError)

    def test_details_default_empty(self):
        err = ToolValidationError("validation failed")
        assert err.details == []

    def test_details_with_structured_info(self):
        details = [
            {"field": "name", "message": "required", "type": "missing"},
            {"field": "age", "message": "must be int", "type": "type_error"},
        ]
        err = ToolValidationError("validation failed", details=details)
        assert len(err.details) == 2
        assert err.details[0]["field"] == "name"
        assert err.details[1]["message"] == "must be int"


class TestToolExecutionError:
    def test_inherits_from_tool_error(self):
        assert issubclass(ToolExecutionError, ToolError)

    def test_tool_name_default_empty(self):
        err = ToolExecutionError("exec failed")
        assert err.tool_name == ""

    def test_tool_name_set(self):
        err = ToolExecutionError("exec failed", tool_name="my_tool")
        assert err.tool_name == "my_tool"


class TestToolPermissionError:
    def test_inherits_from_tool_error(self):
        assert issubclass(ToolPermissionError, ToolError)

    def test_instantiation(self):
        err = ToolPermissionError("permission denied")
        assert str(err) == "permission denied"


class TestToolTimeoutError:
    def test_inherits_from_tool_error(self):
        assert issubclass(ToolTimeoutError, ToolError)

    def test_timeout_default_zero(self):
        err = ToolTimeoutError("timed out")
        assert err.timeout == 0.0

    def test_timeout_set(self):
        err = ToolTimeoutError("timed out", timeout=30.0)
        assert err.timeout == 30.0


class TestExceptionHierarchy:
    def test_all_are_tool_errors(self):
        exceptions = [
            ToolNotFoundError,
            ToolValidationError,
            ToolExecutionError,
            ToolPermissionError,
            ToolTimeoutError,
        ]
        for exc_cls in exceptions:
            assert issubclass(exc_cls, ToolError)

    def test_all_are_catchable_as_exception(self):
        exceptions = [
            ToolError("base"),
            ToolNotFoundError("not found"),
            ToolValidationError("validation"),
            ToolExecutionError("execution"),
            ToolPermissionError("permission"),
            ToolTimeoutError("timeout"),
        ]
        for exc in exceptions:
            assert isinstance(exc, Exception)
