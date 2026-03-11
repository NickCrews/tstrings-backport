import re
import sys
import textwrap
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

# Regex to parse the content inside an interpolation block {content}.
# It captures:
# 1. The main expression.
# 2. An optional debug specifier (=).
# 3. An optional conversion specifier (!r, !s, or !a).
# 4. An optional format specifier (:...).
INTERP_CONTENT_RE = re.compile(
    r"""
    ^
    (?P<expression>.+?)
    (?P<debug>=)?
    (?P<conversion>![rsa])?
    (?P<format_spec>:.+)?
    $
    """,
    re.VERBOSE | re.DOTALL,
)


@dataclass(frozen=True)
class Interpolation:
    """
    Emulates the string.templatelib.Interpolation class from PEP 750.
    Represents an expression inside a template string.
    """

    value: object
    expression: str
    conversion: Optional[Literal["a", "r", "s"]] = None
    format_spec: str = ""


@dataclass(frozen=True)
class Template:
    """
    Emulates the string.templatelib.Template class from PEP 750.
    Represents a parsed t-string literal.
    """

    strings: Tuple[str, ...]
    interpolations: Tuple[Interpolation, ...]


def _split_template(template_string: str):
    """
    Split a template string into alternating static text and interpolation
    content parts.

    Handles ``{{`` and ``}}`` as escape sequences for literal ``{`` and ``}``
    characters (same as Python f-string / PEP 750 t-string rules).
    Handles nested braces within interpolation expressions.

    Returns a list of ``('static', text)`` or ``('interp', content)`` tuples.
    Always starts and ends with a ``'static'`` entry (which may be empty), so
    the returned list has an odd length.
    """
    result = []
    i = 0
    n = len(template_string)
    static_buf: list = []

    while i < n:
        ch = template_string[i]

        if ch == "{":
            if i + 1 < n and template_string[i + 1] == "{":
                # {{ -> literal {
                static_buf.append("{")
                i += 2
            else:
                # Start of an interpolation - flush the static buffer first.
                result.append(("static", "".join(static_buf)))
                static_buf = []

                # Find the matching closing brace, tracking depth for nested
                # braces (e.g. dict literals or set comprehensions inside {}).
                depth = 1
                j = i + 1
                while j < n:
                    if template_string[j] == "{":
                        depth += 1
                    elif template_string[j] == "}":
                        depth -= 1
                        if depth == 0:
                            break
                    j += 1

                if depth != 0:
                    raise SyntaxError("t-string: '{' was never closed")

                content = template_string[i + 1 : j]
                result.append(("interp", content))
                i = j + 1

        elif ch == "}":
            if i + 1 < n and template_string[i + 1] == "}":
                # }} -> literal }
                static_buf.append("}")
                i += 2
            else:
                raise SyntaxError("t-string: single '}' is not allowed")

        else:
            static_buf.append(ch)
            i += 1

    # Always append the trailing static part (may be empty).
    result.append(("static", "".join(static_buf)))
    return result


def t(template_string: str) -> Template:
    """
    Emulates a PEP 750 t-string literal for Python < 3.14.

    This function parses a string with f-string-like syntax and returns
    a `Template` object, correctly evaluating expressions in the caller's
    scope.

    ``{{`` and ``}}`` are escape sequences that produce literal ``{`` and
    ``}`` characters respectively, exactly as Python 3.14 t-strings (and
    f-strings) specify.

    Args:
        template_string: The string to parse, e.g., "Hello {name!r}".

    Returns:
        A `Template` instance containing the parsed static strings and
        evaluated interpolations.
    """
    # Get the execution frame of the caller to evaluate expressions in their scope.
    # sys._getframe(0) is the frame of t()
    # sys._getframe(1) is the frame of the caller of t()
    caller_frame = sys._getframe(1)
    caller_globals = caller_frame.f_globals
    caller_locals = caller_frame.f_locals

    strings = []
    interpolations = []

    for kind, content in _split_template(template_string):
        if kind == "static":
            strings.append(content)
            continue

        # kind == "interp": parse the content of the interpolation block.
        match = INTERP_CONTENT_RE.match(content)
        if not match:
            raise SyntaxError(
                f"t-string: invalid interpolation syntax: {content!r}"
            )

        groups = match.groupdict()
        expression = groups["expression"]

        # The debug specifier is syntactic sugar. It modifies both the
        # preceding string part and the interpolation itself.
        if groups["debug"]:
            # t'{value=}' becomes t'value={value!r}'
            # t'{value=:fmt}' becomes t'value={value!s:fmt}'

            expr_with_possible_ws = groups["expression"]
            eq_index = expr_with_possible_ws.rfind("=")
            if eq_index != -1:
                expr_for_static = expr_with_possible_ws[: eq_index + 1]
                expr_for_eval = expr_with_possible_ws[:eq_index].strip()
                if expr_for_eval.endswith("="):
                    expr_for_eval = expr_for_eval[:-1].rstrip()
            else:
                expr_for_static = expr_with_possible_ws + "="
                expr_for_eval = expr_with_possible_ws.strip()

            # Prepend 'expression=' to the *current* (preceding) static string.
            strings[-1] += expr_for_static

            if groups["conversion"]:
                raise SyntaxError("f-string: cannot specify both conversion and '='")

            # If a format spec is present, conversion becomes 's'. Otherwise, 'r'.
            conv_char = "s" if groups["format_spec"] else "r"
            expression_to_eval = expr_for_eval
        else:
            conv_char = groups["conversion"][1] if groups["conversion"] else None
            expression_to_eval = groups["expression"]

        fmt_spec = groups["format_spec"][1:] if groups["format_spec"] else ""

        # Dedent multiline expressions for evaluation.
        expr_eval_str = textwrap.dedent(expression_to_eval)

        # Evaluate the expression to get its value using the caller's context.
        try:
            value = eval(expr_eval_str, caller_globals, caller_locals)
        except Exception as e:
            msg = f"Failed to evaluate expression '{expression_to_eval}': {e}"
            raise type(e)(msg) from e

        interpolations.append(
            Interpolation(
                value=value,
                expression=expression_to_eval,
                conversion=conv_char,
                format_spec=fmt_spec,
            )
        )

    return Template(strings=tuple(strings), interpolations=tuple(interpolations))
