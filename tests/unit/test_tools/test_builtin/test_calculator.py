"""Tests for the calculator built-in tool."""

import pytest

from ia_agent_fwk.tools.base import ToolContext
from ia_agent_fwk.tools.builtin.calculator import CalculatorInput, CalculatorOutput, CalculatorTool
from ia_agent_fwk.tools.exceptions import ToolExecutionError


@pytest.fixture
def calculator():
    return CalculatorTool()


@pytest.fixture
def ctx():
    return ToolContext(execution_id="test-calc")


class TestBasicArithmetic:
    async def test_addition(self, calculator, ctx):
        result = await calculator.execute(CalculatorInput(expression="2 + 2"), ctx)
        assert isinstance(result, CalculatorOutput)
        assert result.result == 4.0

    async def test_subtraction(self, calculator, ctx):
        result = await calculator.execute(CalculatorInput(expression="10 - 3"), ctx)
        assert result.result == 7.0

    async def test_multiplication(self, calculator, ctx):
        result = await calculator.execute(CalculatorInput(expression="3 * 4"), ctx)
        assert result.result == 12.0

    async def test_division(self, calculator, ctx):
        result = await calculator.execute(CalculatorInput(expression="10 / 3"), ctx)
        assert abs(result.result - 3.3333333333333335) < 1e-10

    async def test_integer_division(self, calculator, ctx):
        result = await calculator.execute(CalculatorInput(expression="7 // 2"), ctx)
        assert result.result == 3.0

    async def test_modulo(self, calculator, ctx):
        result = await calculator.execute(CalculatorInput(expression="7 % 3"), ctx)
        assert result.result == 1.0


class TestOperatorPrecedence:
    async def test_precedence(self, calculator, ctx):
        result = await calculator.execute(CalculatorInput(expression="2 + 3 * 4"), ctx)
        assert result.result == 14.0

    async def test_parentheses(self, calculator, ctx):
        result = await calculator.execute(CalculatorInput(expression="(2 + 3) * 4"), ctx)
        assert result.result == 20.0


class TestNegativeNumbers:
    async def test_negative(self, calculator, ctx):
        result = await calculator.execute(CalculatorInput(expression="-5 + 3"), ctx)
        assert result.result == -2.0

    async def test_unary_plus(self, calculator, ctx):
        result = await calculator.execute(CalculatorInput(expression="+5"), ctx)
        assert result.result == 5.0


class TestFloats:
    async def test_float_multiplication(self, calculator, ctx):
        result = await calculator.execute(CalculatorInput(expression="1.5 * 2"), ctx)
        assert result.result == 3.0


class TestExponentiation:
    async def test_power(self, calculator, ctx):
        result = await calculator.execute(CalculatorInput(expression="2 ** 10"), ctx)
        assert result.result == 1024.0

    async def test_complex_expression(self, calculator, ctx):
        result = await calculator.execute(CalculatorInput(expression="2 ** 10 + 3 * (4 - 1)"), ctx)
        assert result.result == 1033.0


class TestFunctions:
    async def test_sqrt(self, calculator, ctx):
        result = await calculator.execute(CalculatorInput(expression="sqrt(16)"), ctx)
        assert result.result == 4.0

    async def test_abs(self, calculator, ctx):
        result = await calculator.execute(CalculatorInput(expression="abs(-5)"), ctx)
        assert result.result == 5.0

    async def test_round(self, calculator, ctx):
        result = await calculator.execute(CalculatorInput(expression="round(3.7)"), ctx)
        assert result.result == 4.0

    async def test_min(self, calculator, ctx):
        result = await calculator.execute(CalculatorInput(expression="min(1, 2, 3)"), ctx)
        assert result.result == 1.0

    async def test_max(self, calculator, ctx):
        result = await calculator.execute(CalculatorInput(expression="max(1, 2, 3)"), ctx)
        assert result.result == 3.0


class TestDivisionByZero:
    async def test_division_by_zero(self, calculator, ctx):
        with pytest.raises(ToolExecutionError, match="Division by zero"):
            await calculator.execute(CalculatorInput(expression="1 / 0"), ctx)


class TestSecurityRejections:
    async def test_import_rejection(self, calculator, ctx):
        with pytest.raises(ToolExecutionError):
            await calculator.execute(CalculatorInput(expression="__import__('os').system('rm -rf /')"), ctx)

    async def test_eval_rejection(self, calculator, ctx):
        with pytest.raises(ToolExecutionError):
            await calculator.execute(CalculatorInput(expression='eval("1")'), ctx)

    async def test_exec_rejection(self, calculator, ctx):
        with pytest.raises(ToolExecutionError):
            await calculator.execute(CalculatorInput(expression='exec("1")'), ctx)

    async def test_open_rejection(self, calculator, ctx):
        with pytest.raises(ToolExecutionError):
            await calculator.execute(CalculatorInput(expression='open("file")'), ctx)

    async def test_class_traversal_rejection(self, calculator, ctx):
        with pytest.raises(ToolExecutionError):
            await calculator.execute(CalculatorInput(expression="().__class__.__bases__[0].__subclasses__()"), ctx)

    async def test_attribute_access_rejection(self, calculator, ctx):
        with pytest.raises(ToolExecutionError):
            await calculator.execute(CalculatorInput(expression="foo.bar"), ctx)

    async def test_lambda_rejection(self, calculator, ctx):
        with pytest.raises(ToolExecutionError):
            await calculator.execute(CalculatorInput(expression="lambda: 1"), ctx)

    async def test_list_comprehension_rejection(self, calculator, ctx):
        with pytest.raises(ToolExecutionError):
            await calculator.execute(CalculatorInput(expression="[x for x in range(10)]"), ctx)

    async def test_import_statement_rejection(self, calculator, ctx):
        with pytest.raises(ToolExecutionError):
            await calculator.execute(CalculatorInput(expression="import os"), ctx)

    async def test_string_literal_rejection(self, calculator, ctx):
        with pytest.raises(ToolExecutionError):
            await calculator.execute(CalculatorInput(expression='"hello"'), ctx)


class TestOverflowProtection:
    async def test_large_exponent_rejected(self, calculator, ctx):
        with pytest.raises(ToolExecutionError, match="Exponent too large"):
            await calculator.execute(CalculatorInput(expression="10 ** 10000"), ctx)


class TestEmptyExpression:
    async def test_empty_expression(self, calculator, ctx):
        with pytest.raises(ToolExecutionError, match="Empty expression"):
            await calculator.execute(CalculatorInput(expression=""), ctx)

    async def test_whitespace_expression(self, calculator, ctx):
        with pytest.raises(ToolExecutionError, match="Empty expression"):
            await calculator.execute(CalculatorInput(expression="   "), ctx)


class TestToolProperties:
    def test_name(self, calculator):
        assert calculator.name == "calculator"

    def test_description(self, calculator):
        assert "mathematical" in calculator.description.lower()

    def test_tags(self, calculator):
        assert "math" in calculator.tags
        assert "builtin" in calculator.tags
