"""Tools the agent can call. Each is deterministic, sandboxed and validates its own inputs."""
from __future__ import annotations

import ast
import math
import operator
import random
import re
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta


class ToolError(Exception):
    """Raised for bad tool input; the message is shown to the agent so it can recover."""


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, str]
    fn: Callable[..., str]

    def __call__(self, **kwargs) -> str:
        missing = set(self.parameters) - set(kwargs)
        extra = set(kwargs) - set(self.parameters)
        if missing or extra:
            raise ToolError(f"{self.name} expects arguments {sorted(self.parameters)}; "
                            f"missing={sorted(missing)} unexpected={sorted(extra)}")
        return self.fn(**kwargs)


# ------------------------------------------------------------------ calculator
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.Mod: operator.mod, ast.FloorDiv: operator.floordiv,
        ast.USub: operator.neg, ast.UAdd: operator.pos}
_FUNCS = {"sqrt": math.sqrt, "log": math.log, "log10": math.log10, "exp": math.exp, "abs": abs,
          "round": round, "min": min, "max": max, "floor": math.floor, "ceil": math.ceil}


def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ToolError("exponent too large")
        return _OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS:
        return _FUNCS[node.func.id](*[_eval(a) for a in node.args])
    raise ToolError(f"unsupported expression element: {ast.dump(node)[:60]}")


def calculator(expression: str) -> str:
    expr = str(expression).replace(",", "").replace("^", "**").replace("×", "*").replace("%", "/100")
    try:
        value = _eval(ast.parse(expr, mode="eval").body)
    except ToolError:
        raise
    except Exception as e:  # syntax errors, division by zero, ...
        raise ToolError(f"could not evaluate '{expression}': {e}") from e
    return str(round(value, 6)) if isinstance(value, float) else str(value)


# ------------------------------------------------------------------ SQL over a demo shop database
COUNTRIES = ["UK", "Germany", "France", "Spain", "Italy", "Netherlands"]
CATEGORIES = {"Electronics": (40, 400), "Books": (8, 40), "Home": (15, 120), "Sports": (20, 200)}


def build_shop_db(seed: int = 7) -> sqlite3.Connection:
    rng = random.Random(seed)
    con = sqlite3.connect(":memory:", check_same_thread=False)
    con.executescript("""
        CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT, country TEXT, signup_date TEXT);
        CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT, category TEXT, price REAL);
        CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER, product_id INTEGER,
                             quantity INTEGER, order_date TEXT, status TEXT);""")
    first = ["Ava", "Ben", "Chloe", "Dan", "Ella", "Finn", "Grace", "Hugo", "Isla", "Jack"]
    last = ["Smith", "Muller", "Martin", "Garcia", "Rossi", "de Vries", "Brown", "Klein"]
    for i in range(1, 61):
        d = date(2025, 1, 1) + timedelta(days=rng.randint(0, 364))
        con.execute("INSERT INTO customers VALUES (?,?,?,?)",
                    (i, f"{rng.choice(first)} {rng.choice(last)}", rng.choices(COUNTRIES, [5, 3, 2, 2, 1, 1])[0],
                     d.isoformat()))
    pid = 0
    for cat, (lo, hi) in CATEGORIES.items():
        for j in range(5):
            pid += 1
            con.execute("INSERT INTO products VALUES (?,?,?,?)",
                        (pid, f"{cat} item {j + 1}", cat, round(rng.uniform(lo, hi), 2)))
    for i in range(1, 501):
        d = date(2026, 1, 1) + timedelta(days=rng.randint(0, 180))
        con.execute("INSERT INTO orders VALUES (?,?,?,?,?,?)",
                    (i, rng.randint(1, 60), rng.randint(1, pid), rng.randint(1, 4), d.isoformat(),
                     rng.choices(["completed", "refunded", "pending"], [85, 8, 7])[0]))
    con.commit()
    return con


SCHEMA = ("customers(id, name, country, signup_date YYYY-MM-DD); products(id, name, category, price GBP); "
          "orders(id, customer_id, product_id, quantity, order_date YYYY-MM-DD, status in "
          "'completed'|'refunded'|'pending'). Revenue of an order = quantity * products.price.")


def make_sql_tool(con: sqlite3.Connection) -> Callable[[str], str]:
    def sql_query(query: str) -> str:
        q = str(query).strip().rstrip(";")
        if not re.match(r"(?is)^\s*(select|with)\b", q) or re.search(
                r"(?i)\b(insert|update|delete|drop|alter|create|attach|pragma)\b", q):
            raise ToolError("only read-only SELECT queries are allowed")
        try:
            cur = con.execute(q)
        except sqlite3.Error as e:
            raise ToolError(f"SQL error: {e}") from e
        cols = [c[0] for c in cur.description]
        rows = cur.fetchmany(20)
        lines = [" | ".join(cols)] + [" | ".join(str(v) for v in r) for r in rows]
        return "\n".join(lines) + ("\n... (truncated to 20 rows)" if len(rows) == 20 else "")
    return sql_query


# ------------------------------------------------------------------ unit / currency conversion
_LINEAR = {  # factor to SI base unit
    "m": ("length", 1.0), "km": ("length", 1000.0), "cm": ("length", 0.01), "mm": ("length", 0.001),
    "mile": ("length", 1609.344), "yard": ("length", 0.9144), "foot": ("length", 0.3048),
    "inch": ("length", 0.0254), "kg": ("mass", 1.0), "g": ("mass", 0.001), "lb": ("mass", 0.45359237),
    "oz": ("mass", 0.028349523125), "stone": ("mass", 6.35029318), "l": ("volume", 1.0),
    "ml": ("volume", 0.001), "gallon_uk": ("volume", 4.54609), "gallon_us": ("volume", 3.785411784),
    "gbp": ("currency", 1.0), "usd": ("currency", 1 / 1.27), "eur": ("currency", 1 / 1.17),
}
_ALIASES = {"miles": "mile", "meters": "m", "metres": "m", "kilometers": "km", "kilometres": "km",
            "feet": "foot", "ft": "foot", "inches": "inch", "in": "inch", "pounds": "lb", "lbs": "lb",
            "kilograms": "kg", "grams": "g", "litres": "l", "liters": "l", "gallon": "gallon_uk",
            "celsius": "c", "fahrenheit": "f", "kelvin": "k", "°c": "c", "°f": "f", "£": "gbp", "$": "usd", "€": "eur"}


def _unit(u: str) -> str:
    u = str(u).strip().lower()
    return _ALIASES.get(u, u)


def convert_units(value: float, from_unit: str, to_unit: str) -> str:
    try:
        v = float(str(value).replace(",", ""))
    except ValueError as e:
        raise ToolError(f"value must be a number, got {value!r}") from e
    f, t = _unit(from_unit), _unit(to_unit)
    temps = {"c": (1.0, 0.0), "f": (5 / 9, -32.0), "k": (1.0, -273.15)}
    if f in temps and t in temps:
        sc, off = temps[f]
        c = (v + off) * sc if f == "f" else v + off
        out = c * 9 / 5 + 32 if t == "f" else c - temps[t][1] if t == "k" else c
        return f"{round(out, 4)} {to_unit}"
    if f not in _LINEAR or t not in _LINEAR:
        raise ToolError(f"unknown unit; supported: {sorted(_LINEAR)} + c/f/k")
    if _LINEAR[f][0] != _LINEAR[t][0]:
        raise ToolError(f"cannot convert {_LINEAR[f][0]} to {_LINEAR[t][0]}")
    return f"{round(v * _LINEAR[f][1] / _LINEAR[t][1], 4)} {to_unit}"


# ------------------------------------------------------------------ policy knowledge base
POLICIES = {
    "returns": "Customers may return any unused item within 30 days of delivery for a full refund. "
               "Electronics must be returned within 14 days. Refunds are issued within 5 business days.",
    "shipping": "Standard UK shipping costs £3.99 and is free for orders over £50. Express shipping costs £9.99. "
                "International shipping to the EU costs £12.50.",
    "warranty": "Electronics carry a 2-year manufacturer warranty. Other products carry a 1-year warranty.",
    "loyalty": "Customers earn 1 loyalty point per £1 spent. 100 points can be redeemed for a £5 voucher.",
}


def search_policies(query: str) -> str:
    words = set(re.findall(r"[a-z]+", str(query).lower()))
    scored = sorted(((len(words & set(re.findall(r"[a-z]+", (k + " " + v).lower()))), k, v)
                     for k, v in POLICIES.items()), reverse=True)
    hits = [f"[{k}] {v}" for s, k, v in scored if s > 0][:2]
    return "\n".join(hits) if hits else "No matching policy found."


def default_tools(con: sqlite3.Connection | None = None) -> dict[str, Tool]:
    con = con or build_shop_db()
    tools = [
        Tool("calculator", "Evaluate an arithmetic expression, e.g. '(1250*0.175)+sqrt(16)'.",
             {"expression": "string"}, calculator),
        Tool("sql_query", f"Run a read-only SQLite SELECT on the shop database. Schema: {SCHEMA}",
             {"query": "string"}, make_sql_tool(con)),
        Tool("convert_units", "Convert a value between units (length, mass, volume, temperature c/f/k) "
             "or currencies gbp/usd/eur at fixed reference rates (1 GBP = 1.27 USD = 1.17 EUR).",
             {"value": "number", "from_unit": "string", "to_unit": "string"}, convert_units),
        Tool("search_policies", "Search the company policy handbook (returns, shipping, warranty, loyalty).",
             {"query": "string"}, search_policies),
    ]
    return {t.name: t for t in tools}
