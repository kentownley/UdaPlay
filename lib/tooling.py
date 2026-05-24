"""Lightweight tool-decoration framework.

Provides the ``@tool`` decorator used to expose plain Python functions as
OpenAI-compatible tools. A decorated function is wrapped in a ``Tool`` object
that:

  * is still directly callable (``tool_fn(query="...")``),
  * exposes ``.name``, ``.description``, ``.parameters``,
  * exposes ``.schema`` — the dict the OpenAI Chat Completions API expects in
    its ``tools=[...]`` field.

Parameter schemas are inferred from the function signature + type hints.
The docstring becomes the tool description.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, get_args, get_origin, get_type_hints


_PRIMITIVE_TO_JSON = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
}


def _annotation_to_json_schema(annotation: Any) -> dict[str, Any]:
    if annotation in _PRIMITIVE_TO_JSON:
        return {"type": _PRIMITIVE_TO_JSON[annotation]}

    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin in (list, tuple):
        item_type = args[0] if args else str
        return {"type": "array", "items": _annotation_to_json_schema(item_type)}
    if origin is dict:
        return {"type": "object"}
    if annotation is dict or annotation is list:
        return {"type": "object" if annotation is dict else "array"}

    # Fallback: treat unknown / Any / typing.Optional[...] without further drilling.
    return {"type": "string"}


class Tool:
    """A callable wrapper that also carries an OpenAI tool schema."""

    def __init__(
        self,
        fn: Callable[..., Any],
        name: str | None = None,
        description: str | None = None,
    ) -> None:
        self.fn = fn
        self.name = name or fn.__name__
        self.description = (description or fn.__doc__ or "").strip()
        self.parameters = self._build_parameters(fn)

    @staticmethod
    def _build_parameters(fn: Callable[..., Any]) -> dict[str, Any]:
        sig = inspect.signature(fn)
        try:
            hints = get_type_hints(fn)
        except Exception:
            hints = {}

        properties: dict[str, Any] = {}
        required: list[str] = []

        for pname, param in sig.parameters.items():
            if pname in ("self", "cls"):
                continue
            annotation = hints.get(pname, str)
            schema = _annotation_to_json_schema(annotation)
            properties[pname] = schema
            if param.default is inspect.Parameter.empty:
                required.append(pname)

        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.fn(*args, **kwargs)

    def __repr__(self) -> str:
        return f"Tool(name={self.name!r})"


def tool(
    fn: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
) -> Tool | Callable[[Callable[..., Any]], Tool]:
    """Decorator: turn a plain function into a ``Tool``.

    Usage::

        @tool
        def retrieve_game(query: str, n_results: int = 5) -> dict:
            \"\"\"Search the catalog.\"\"\"
            ...

        @tool(name="search", description="...")
        def search(...): ...
    """

    def wrap(f: Callable[..., Any]) -> Tool:
        return Tool(f, name=name, description=description)

    if fn is None:
        return wrap
    return wrap(fn)
