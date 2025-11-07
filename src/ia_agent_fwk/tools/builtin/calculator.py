"""Calculator built-in tool with safe AST-walking evaluator.

Evaluates mathematical expressions without using ``eval()`` or ``exec()``.
Uses a custom AST walker that only processes explicitly allowed node types.
"""

from __future__ import annotations

import ast
import math
import operator
from typing import Any

from pydantic import BaseModel, ConfigDict

from ia_agent_fwk.tools.base import Tool, ToolContext
from ia_agent_fwk.tools.exceptions import ToolExecutionError

# Maximum recursion depth for nested expressions
_MAX_DEPTH = 50

# Maximum allowed exponent to prevent numeric overflow
_MAX_EXPONENT = 1000

# Allowed binary operators
_BINARY_OPS: dict[type[ast.operator], Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

# Allowed unary operators
_UNARY_OPS: dict[type[ast.unaryop], Any] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

# Allowed function names and their implementations
_ALLOWED_FUNCTIONS: dict[str, Any] = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sqrt": math.sqrt,
}


class CalculatorInput(BaseModel):
    """Input schema for the calculator tool."""

    model_config = ConfigDict(frozen=True)

    expression: str


class CalculatorOutput(BaseModel):
    """Output schema for the calculator tool."""

    model_config = ConfigDict(frozen=True)

    result: float
    expression: str


class SafeEvaluator(ast.NodeVisitor):
    """AST-walking evaluator that only processes allowed node types.

    Uses an allowlist-only approach: any node type not explicitly handled
    is rejected with an error.
    """

    def __init__(self) -> None:
        self._depth = 0

    def evaluate(self, expression: str) -> float:
        """Parse and evaluate a mathematical expression.

        Parameters
        ----------
        expression:
            The mathematical expression string.

        Returns
        -------
        float
            The numeric result.

        Raises
        ------
        ToolExecutionError
            If the expression is invalid, unsafe, or causes an error.

        """
        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError as exc:
            msg = f"Invalid expression syntax: {exc}"
            raise ToolExecutionError(msg, tool_name="calculator") from exc

        try:
            result = self.visit(tree.body)
        except ToolExecutionError:
            raise
        except Exception as exc:
            msg = f"Error evaluating expression: {type(exc).__name__}: {exc}"
            raise ToolExecutionError(msg, tool_name="calculator") from exc

        if isinstance(result, complex):
            msg = "Complex numbers are not supported."
            raise ToolExecutionError(msg, tool_name="calculator")

        return float(result)

    def visit(self, node: ast.AST) -> Any:
        """Visit an AST node, enforcing depth limits."""
        self._depth += 1
        if self._depth > _MAX_DEPTH:
            msg = f"Expression too deeply nested (max depth: {_MAX_DEPTH})."
            raise ToolExecutionError(msg, tool_name="calculator")
        try:
            result = super().visit(node)
        finally:
            self._depth -= 1
        return result

    def visit_Constant(self, node: ast.Constant) -> float:
        """Handle numeric constants."""
        # Reject booleans explicitly (bool is a subclass of int in Python)
        if isinstance(node.value, bool):
            msg = "Boolean constants are not allowed. Only numeric expressions are supported."
            raise ToolExecutionError(msg, tool_name="calculator")
        if isinstance(node.value, (int, float)):
            return float(node.value)
        msg = f"Unsupported constant type: {type(node.value).__name__}. Only numbers are allowed."
        raise ToolExecutionError(msg, tool_name="calculator")

    def visit_BinOp(self, node: ast.BinOp) -> float:
        """Handle binary operations (+, -, *, /, //, %, **)."""
        op_type = type(node.op)
        if op_type not in _BINARY_OPS:
            msg = f"Unsupported operator: {op_type.__name__}."
            raise ToolExecutionError(msg, tool_name="calculator")

        left = self.visit(node.left)
        right = self.visit(node.right)

        # Exponent overflow protection
        if op_type is ast.Pow and isinstance(right, (int, float)) and abs(right) > _MAX_EXPONENT:
            msg = f"Exponent too large: {right}. Maximum allowed: {_MAX_EXPONENT}."
            raise ToolExecutionError(msg, tool_name="calculator")

        op_func = _BINARY_OPS[op_type]

        try:
            return float(op_func(left, right))
        except ZeroDivisionError as exc:
            msg = "Division by zero."
            raise ToolExecutionError(msg, tool_name="calculator") from exc
        except OverflowError as exc:
            msg = "Numeric overflow."
            raise ToolExecutionError(msg, tool_name="calculator") from exc

    def visit_UnaryOp(self, node: ast.UnaryOp) -> float:
        """Handle unary operations (+, -)."""
        op_type = type(node.op)
        if op_type not in _UNARY_OPS:
            msg = f"Unsupported unary operator: {op_type.__name__}."
            raise ToolExecutionError(msg, tool_name="calculator")
        operand = self.visit(node.operand)
        return float(_UNARY_OPS[op_type](operand))

    def visit_Call(self, node: ast.Call) -> float:
        """Handle function calls (only whitelisted functions)."""
        if not isinstance(node.func, ast.Name):
            msg = "Only simple function calls are allowed (no method calls or attribute access)."
            raise ToolExecutionError(msg, tool_name="calculator")

        func_name = node.func.id
        if func_name not in _ALLOWED_FUNCTIONS:
            msg = f"Function '{func_name}' is not allowed. Allowed: {', '.join(sorted(_ALLOWED_FUNCTIONS))}."
            raise ToolExecutionError(msg, tool_name="calculator")

        if node.keywords:
            msg = "Keyword arguments are not allowed in function calls."
            raise ToolExecutionError(msg, tool_name="calculator")

        args = [self.visit(arg) for arg in node.args]

        try:
            return float(_ALLOWED_FUNCTIONS[func_name](*args))
        except (ValueError, TypeError, ZeroDivisionError, OverflowError) as exc:
            msg = f"Error in function '{func_name}': {exc}"
            raise ToolExecutionError(msg, tool_name="calculator") from exc

    def visit_Name(self, node: ast.Name) -> float:
        """Reject variable names (not allowed)."""
        msg = f"Variable '{node.id}' is not allowed. Only numeric expressions and whitelisted functions are supported."
        raise ToolExecutionError(msg, tool_name="calculator")

    def generic_visit(self, node: ast.AST) -> Any:
        """Reject any unhandled node type."""
        msg = f"Unsupported expression element: {type(node).__name__}. Only arithmetic expressions are allowed."
        raise ToolExecutionError(msg, tool_name="calculator")


class CalculatorTool(Tool):
    """Calculator tool that safely evaluates mathematical expressions.

    Uses an AST-walking evaluator. Does NOT use ``eval()`` or ``exec()``.
    Supports: ``+``, ``-``, ``*``, ``/``, ``//``, ``%``, ``**``,
    parentheses, unary ``+``/``-``, and whitelisted functions
    (``abs``, ``round``, ``min``, ``max``, ``sqrt``).
    """

    @property
    def name(self) -> str:
        return "calculator"

    @property
    def description(self) -> str:
        return (
            "Evaluate mathematical expressions safely. "
            "Supports arithmetic operators (+, -, *, /, //, %, **), "
            "parentheses, and functions (abs, round, min, max, sqrt)."
        )

    @property
    def input_schema(self) -> type[BaseModel]:
        return CalculatorInput

    @property
    def output_schema(self) -> type[BaseModel]:
        return CalculatorOutput

    @property
    def tags(self) -> list[str]:
        return ["math", "builtin"]

    async def execute(self, validated_input: BaseModel, context: ToolContext) -> BaseModel:  # noqa: ARG002
        """Evaluate the mathematical expression."""
        assert isinstance(validated_input, CalculatorInput)  # noqa: S101
        expression = validated_input.expression.strip()

        if not expression:
            msg = "Empty expression."
            raise ToolExecutionError(msg, tool_name="calculator")

        evaluator = SafeEvaluator()
        result = evaluator.evaluate(expression)
        return CalculatorOutput(result=result, expression=expression)
