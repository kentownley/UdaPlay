"""State-machine agent with per-session conversational memory.

The agent runs the canonical loop:

    START -> REASON -> (TOOL_CALL -> REASON)* -> ANSWER -> END

bounded by ``max_iters``. Conversation history is keyed by ``session_id`` so
follow-up questions in the same session can resolve pronouns ("it", "its
publisher") against prior turns.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .tooling import Tool


class State(str, Enum):
    START = "START"
    REASON = "REASON"
    TOOL_CALL = "TOOL_CALL"
    ANSWER = "ANSWER"
    END = "END"


@dataclass
class Step:
    state: State
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class Run:
    """Trace of a single agent invocation, for inspection / reporting."""

    query: str
    session_id: str | None
    steps: list[Step] = field(default_factory=list)
    final_answer: str | None = None
    iterations: int = 0

    @property
    def tool_calls(self) -> list[dict[str, Any]]:
        return [s.payload for s in self.steps if s.state is State.TOOL_CALL]

    @property
    def used_web_search(self) -> bool:
        return any(c.get("name") == "game_web_search" for c in self.tool_calls)

    def to_report(self) -> str:
        lines = [f"## Question\n{self.query}", "", "## Answer",
                 self.final_answer or "(no answer)"]
        lines += ["", "## Reasoning trace"]
        for i, call in enumerate(self.tool_calls, 1):
            preview = json.dumps(call.get("arguments", {}), default=str)[:200]
            lines.append(f"{i}. **{call['name']}** — args: `{preview}`")
        lines += ["",
                  f"_Iterations: {self.iterations} • "
                  f"Used web search: {self.used_web_search} • "
                  f"Session: {self.session_id or '(stateless)'}_"]
        return "\n".join(lines)


class Agent:
    """Tool-using agent driven by OpenAI function calling.

    Parameters
    ----------
    model_name : str
        OpenAI chat model id (e.g. ``"gpt-4o-mini"``).
    instructions : str
        System prompt installed at the top of every session.
    tools : list[Tool]
        Tools decorated with ``@tool`` from ``lib.tooling``.
    openai_client : openai.OpenAI
        Authenticated OpenAI client.
    max_iters : int
        Cap on REASON/TOOL_CALL cycles per invocation.
    """

    def __init__(
        self,
        model_name: str,
        instructions: str,
        tools: list[Tool],
        openai_client: Any,
        max_iters: int = 8,
    ) -> None:
        self.model_name = model_name
        self.instructions = instructions
        self.tools: dict[str, Tool] = {t.name: t for t in tools}
        self.tool_schemas = [t.schema for t in tools]
        self.client = openai_client
        self.max_iters = max_iters
        self._sessions: dict[str, list[dict[str, Any]]] = {}

    # ----- session API -----

    def _session_messages(self, session_id: str | None) -> list[dict[str, Any]]:
        """Return the message list to mutate for this invocation.

        Stateless calls (``session_id is None``) get a fresh transcript that is
        discarded after the run. Stateful calls share a transcript across runs.
        """
        if session_id is None:
            return [{"role": "system", "content": self.instructions}]
        if session_id not in self._sessions:
            self._sessions[session_id] = [
                {"role": "system", "content": self.instructions}
            ]
        return self._sessions[session_id]

    def reset_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def get_history(self, session_id: str) -> list[dict[str, Any]]:
        return list(self._sessions.get(session_id, []))

    # ----- main loop -----

    def invoke(self, query: str, session_id: str | None = None) -> Run:
        messages = self._session_messages(session_id)
        messages.append({"role": "user", "content": query})

        run = Run(query=query, session_id=session_id)
        run.steps.append(Step(State.START, {"query": query}))

        for i in range(self.max_iters):
            run.iterations = i + 1
            run.steps.append(Step(State.REASON, {"iteration": i + 1}))

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                tools=self.tool_schemas,
                tool_choice="auto",
                temperature=0,
            )
            msg = response.choices[0].message
            messages.append(msg.model_dump(exclude_none=True))

            if not msg.tool_calls:
                run.final_answer = msg.content
                run.steps.append(Step(State.ANSWER, {"content": msg.content}))
                run.steps.append(Step(State.END, {}))
                return run

            for tc in msg.tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}

                tool_obj = self.tools.get(name)
                if tool_obj is None:
                    result: Any = {"error": f"Unknown tool: {name}"}
                else:
                    try:
                        result = tool_obj(**args)
                    except Exception as e:  # noqa: BLE001
                        result = {"error": f"{type(e).__name__}: {e}"}

                # Pydantic objects → dicts before serializing.
                if hasattr(result, "model_dump"):
                    serializable = result.model_dump()
                else:
                    serializable = result

                run.steps.append(
                    Step(
                        State.TOOL_CALL,
                        {"name": name, "arguments": args, "result": serializable},
                    )
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": name,
                        "content": json.dumps(serializable, default=str),
                    }
                )

        run.final_answer = (
            "I hit the max-iterations limit before reaching a confident answer."
        )
        run.steps.append(Step(State.END, {"reason": "max_iters"}))
        return run
