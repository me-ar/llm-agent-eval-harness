"""Deterministic tests: tools, parser, agent loop (scripted LLM) and the task suite's ground truth."""
import pytest

from agentkit.agent import Agent, parse_reply
from agentkit.harness import classify_failure, evaluate, summarise
from agentkit.llm import ScriptedLLM
from agentkit.tasks import build_tasks, num_check, numbers_in
from agentkit.tools import ToolError, build_shop_db, calculator, convert_units, default_tools, search_policies

CON = build_shop_db()
TOOLS = default_tools(CON)


def test_calculator_is_safe_and_correct():
    assert calculator("(1234.5*3.7) - 987.65/4.3") == str(round(1234.5 * 3.7 - 987.65 / 4.3, 6))
    assert calculator("2^10") == "1024" and calculator("sqrt(98596)") == "314.0"
    for bad in ["__import__('os')", "open('x')", "9**9999", "1/0"]:
        with pytest.raises(ToolError):
            calculator(bad)


def test_sql_tool_is_read_only():
    assert TOOLS["sql_query"](query="SELECT COUNT(*) AS n FROM customers").splitlines() == ["n", "60"]
    for bad in ["DROP TABLE orders", "DELETE FROM orders", "SELECT 1; DROP TABLE orders"]:
        with pytest.raises(ToolError):
            TOOLS["sql_query"](query=bad)


def test_unit_conversion():
    assert convert_units(98.6, "fahrenheit", "celsius").startswith("37.0")
    assert convert_units(26.2, "miles", "km").startswith("42.16")
    assert convert_units(100, "gbp", "usd").startswith("127.0")
    with pytest.raises(ToolError):
        convert_units(1, "kg", "km")


def test_tool_argument_validation():
    with pytest.raises(ToolError, match="missing"):
        TOOLS["convert_units"](value=1, from_unit="kg")


def test_policy_search():
    assert "14 days" in search_policies("return electronics")


@pytest.mark.parametrize("text", ['{"final_answer": "4"}', 'Sure!\n```json\n{"final_answer": "4"}\n```',
                                  'thinking {not json} then {"final_answer": "4", "x": {"y": "}"}}'])
def test_parse_reply_is_robust(text):
    assert parse_reply(text)["final_answer"] == "4"


def test_agent_uses_tool_then_answers():
    llm = ScriptedLLM(['{"thought": "calc", "tool": "calculator", "args": {"expression": "4837*2569"}}',
                       '{"thought": "done", "final_answer": "12,426,253"}'])
    tr = Agent(llm, TOOLS).run("t", "What is 4837*2569?")
    assert tr.stop_reason == "final_answer" and tr.steps[0].observation == "12426253"
    assert "Observation from calculator" in llm.calls[1][-1]["content"]


def test_agent_recovers_from_bad_format_and_tool_errors():
    llm = ScriptedLLM(["I think the answer is 5", '{"tool": "nope", "args": {}}',
                       '{"tool": "calculator", "args": {"expression": "1/0"}}', '{"final_answer": "5"}'])
    tr = Agent(llm, TOOLS).run("t", "q")
    assert tr.final_answer == "5" and tr.n_invalid == 2 and tr.n_tool_errors == 1
    assert classify_failure(tr, False) == "tool error not recovered"


def test_agent_stops_at_max_steps():
    llm = ScriptedLLM(['{"tool": "calculator", "args": {"expression": "1+1"}}'] * 10)
    tr = Agent(llm, TOOLS, max_steps=3).run("t", "q")
    assert tr.stop_reason == "max_steps" and len(tr.steps) == 3


def test_task_suite_ground_truth_is_consistent():
    tasks = build_tasks(CON)
    assert len(tasks) == 30 and len({t.id for t in tasks}) == 30
    assert {t.category for t in tasks} == {"arithmetic", "sql", "units", "policy", "multi_tool"}
    sql = next(t for t in tasks if "refunded'?" in t.question)
    assert sql.check(TOOLS["sql_query"](query="SELECT COUNT(*) FROM orders WHERE status='refunded'"))


def test_answer_checking():
    assert numbers_in("about 1,234.5 units and 7") == [1234.5, 7.0]
    assert num_check(42.16)("42.16 km") and not num_check(42.16)("42 km")


def test_harness_end_to_end_with_scripted_oracle():
    tasks = build_tasks(CON)[:3]
    oracle = ScriptedLLM([f'{{"final_answer": "{t.expected}"}}' for t in tasks])
    df, traces = evaluate(Agent(oracle, TOOLS), tasks, "oracle", log=lambda *_: None)
    assert df["correct"].all() and summarise(df)["accuracy"].iloc[0] == 1.0
