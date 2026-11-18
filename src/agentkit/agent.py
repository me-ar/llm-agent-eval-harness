"""A minimal, transparent ReAct-style agent with a JSON tool-calling protocol.

Each turn the model must reply with ONE JSON object, either
  {"thought": "...", "tool": "<name>", "args": {...}}      -> the tool runs, its output is fed back
  {"thought": "...", "final_answer": "..."}                 -> the episode ends
Malformed replies are not fatal: the parser error is fed back so the model can correct itself.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field

from .llm import LLM
from .tools import Tool, ToolError

SYSTEM = """You are a careful assistant that solves tasks step by step using tools.
Never do arithmetic in your head - use the calculator. Never guess data - query the database.

Available tools:
{tools}

Reply with exactly ONE JSON object and nothing else:
- to call a tool: {{"thought": "<short reasoning>", "tool": "<tool name>", "args": {{...}}}}
- to finish:     {{"thought": "<short reasoning>", "final_answer": "<concise answer>"}}"""

DIRECT_SYSTEM = """You are a helpful assistant. Answer the question as accurately as you can.
Reply with exactly ONE JSON object: {"final_answer": "<concise answer>"}"""


@dataclass
class Step:
    raw: str
    tool: str | None = None
    args: dict | None = None
    observation: str | None = None
    error: str | None = None


@dataclass
class Trace:
    task_id: str
    final_answer: str | None = None
    steps: list[Step] = field(default_factory=list)
    stop_reason: str = "max_steps"           # final_answer | max_steps
    seconds: float = 0.0

    @property
    def n_tool_calls(self) -> int:
        return sum(s.tool is not None for s in self.steps)

    @property
    def n_invalid(self) -> int:
        return sum(s.error is not None and s.error.startswith("format") for s in self.steps)

    @property
    def n_tool_errors(self) -> int:
        return sum(s.error is not None and s.error.startswith("tool") for s in self.steps)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(n_tool_calls=self.n_tool_calls, n_invalid=self.n_invalid, n_tool_errors=self.n_tool_errors)
        return d


def parse_reply(text: str) -> dict:
    """Extract the first balanced JSON object from a model reply (tolerates code fences / chatter)."""
    text = re.sub(r"```(?:json)?", "", text)
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                esc = (ch == "\\") and not esc
                if ch == '"' and not esc:
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start:i + 1])
                        if isinstance(obj, dict):
                            return obj
                    except json.JSONDecodeError:
                        break
                    break
        start = text.find("{", start + 1)
    raise ValueError("no valid JSON object found")


def describe_tools(tools: dict[str, Tool]) -> str:
    return "\n".join(f"- {t.name}({', '.join(f'{k}: {v}' for k, v in t.parameters.items())}): {t.description}"
                     for t in tools.values())


class Agent:
    def __init__(self, llm: LLM, tools: dict[str, Tool], max_steps: int = 6, use_tools: bool = True):
        self.llm, self.tools, self.max_steps, self.use_tools = llm, tools, max_steps, use_tools

    def run(self, task_id: str, question: str) -> Trace:
        trace, t0 = Trace(task_id), time.perf_counter()
        system = SYSTEM.format(tools=describe_tools(self.tools)) if self.use_tools else DIRECT_SYSTEM
        messages = [{"role": "system", "content": system}, {"role": "user", "content": question}]
        for _ in range(self.max_steps if self.use_tools else 1):
            raw = self.llm.chat(messages)
            step = Step(raw=raw)
            trace.steps.append(step)
            messages.append({"role": "assistant", "content": raw})
            try:
                obj = parse_reply(raw)
            except ValueError:
                if not self.use_tools:              # direct mode: accept plain-text answers
                    trace.final_answer, trace.stop_reason = raw.strip(), "final_answer"
                    break
                step.error = "format: reply was not a JSON object"
                messages.append({"role": "user", "content": "Error: reply with ONE valid JSON object as instructed."})
                continue
            if "final_answer" in obj or not self.use_tools:
                trace.final_answer = str(obj.get("final_answer", raw.strip()))
                trace.stop_reason = "final_answer"
                break
            name, args = obj.get("tool"), obj.get("args", {})
            if name not in self.tools or not isinstance(args, dict):
                step.error = f"format: unknown tool {name!r} or args not an object"
                messages.append({"role": "user", "content": f"Error: {step.error}. Tools: {list(self.tools)}"})
                continue
            step.tool, step.args = name, args
            try:
                step.observation = self.tools[name](**args)
            except ToolError as e:
                step.error = f"tool: {e}"
                step.observation = f"Error: {e}"
            except Exception as e:  # never let a tool crash the episode
                step.error = f"tool: unexpected {type(e).__name__}: {e}"
                step.observation = f"Error: {e}"
            messages.append({"role": "user", "content": f"Observation from {name}:\n{step.observation}"})
        trace.seconds = round(time.perf_counter() - t0, 2)
        return trace
