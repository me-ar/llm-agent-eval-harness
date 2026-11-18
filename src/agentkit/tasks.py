"""Evaluation suite: 30 tasks across 5 categories, with programmatic answer checking.

Database answers are computed from the same seeded database the agent queries, so the
ground truth is correct by construction.
"""
from __future__ import annotations

import re
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass

from .tools import build_shop_db

NUM = re.compile(r"-?\d[\d,]*\.?\d*(?:e-?\d+)?")


@dataclass
class Task:
    id: str
    category: str
    question: str
    expected: str | float
    check: Callable[[str], bool]


def numbers_in(text: str) -> list[float]:
    out = []
    for m in NUM.findall(text or ""):
        try:
            out.append(float(m.replace(",", "")))
        except ValueError:
            pass
    return out


def num_check(expected: float, rel: float = 0.0005, abs_tol: float = 0.011) -> Callable[[str], bool]:
    return lambda ans: any(abs(x - expected) <= max(abs_tol, rel * abs(expected)) for x in numbers_in(ans))


def text_check(*needles: str) -> Callable[[str], bool]:
    return lambda ans: all(n.lower() in (ans or "").lower() for n in needles)


def _q(con: sqlite3.Connection, sql: str):
    return con.execute(sql).fetchone()[0]


def build_tasks(con: sqlite3.Connection | None = None) -> list[Task]:
    con = con or build_shop_db()
    T: list[Task] = []

    def add(cat, q, exp, check=None):
        T.append(Task(f"{cat[:4]}-{sum(t.category == cat for t in T) + 1:02d}", cat, q, exp,
                      check or num_check(float(exp))))

    # --- arithmetic: easy to get wrong without a calculator
    add("arithmetic", "What is 4,837 multiplied by 2,569?", 4837 * 2569)
    add("arithmetic", "What is 17.5% of 2,348.60? Give the answer to 2 decimal places.", round(0.175 * 2348.60, 2))
    add("arithmetic", "Compute (1234.5 * 3.7) - (987.65 / 4.3). Round to 2 decimal places.",
        round(1234.5 * 3.7 - 987.65 / 4.3, 2))
    add("arithmetic", "What is the square root of 98,596?", 314)
    add("arithmetic", "A £12,500 loan grows at 4.2% compound interest per year. What is it worth after 7 years? "
        "Round to the nearest penny.", round(12500 * 1.042 ** 7, 2))
    add("arithmetic", "What is 2 to the power of 37?", 2 ** 37)

    # --- SQL over the shop database
    add("sql", "How many orders in the shop database have status 'refunded'?",
        _q(con, "SELECT COUNT(*) FROM orders WHERE status='refunded'"))
    add("sql", "How many customers are based in Germany?", _q(con, "SELECT COUNT(*) FROM customers WHERE country='Germany'"))
    add("sql", "What is the price of the most expensive product?", _q(con, "SELECT MAX(price) FROM products"))
    add("sql", "How many units (sum of quantity) were sold in completed orders?",
        _q(con, "SELECT SUM(quantity) FROM orders WHERE status='completed'"))
    top_country = _q(con, "SELECT country FROM customers GROUP BY country ORDER BY COUNT(*) DESC LIMIT 1")
    add("sql", "Which country has the most customers?", top_country, text_check(top_country))
    add("sql", "What was the total revenue (quantity x price) of completed orders placed in March 2026? "
        "Round to 2 decimal places.",
        round(_q(con, """SELECT SUM(o.quantity*p.price) FROM orders o JOIN products p ON p.id=o.product_id
                         WHERE o.status='completed' AND o.order_date LIKE '2026-03-%'"""), 2))
    top_cat = _q(con, """SELECT p.category FROM orders o JOIN products p ON p.id=o.product_id
                         WHERE o.status='completed' GROUP BY p.category ORDER BY SUM(o.quantity*p.price) DESC LIMIT 1""")
    add("sql", "Which product category generated the most revenue from completed orders?", top_cat, text_check(top_cat))
    add("sql", "How many distinct customers placed at least one order?",
        _q(con, "SELECT COUNT(DISTINCT customer_id) FROM orders"))

    # --- unit conversion
    add("units", "How many kilometres is a marathon of 26.2 miles?", round(26.2 * 1.609344, 2))
    add("units", "Convert 98.6 degrees Fahrenheit to Celsius.", 37.0)
    add("units", "How many pounds (lb) is 72.5 kg? Round to 2 decimals.", round(72.5 / 0.45359237, 2))
    add("units", "Convert 250 GBP to USD using the reference rate.", round(250 * 1.27, 2))
    add("units", "How many US gallons are in 50 litres? Round to 3 decimals.", round(50 / 3.785411784, 3))
    add("units", "How many feet tall is a 3.2 metre wall? Round to 2 decimals.", round(3.2 / 0.3048, 2))

    # --- policy lookup (retrieval)
    add("policy", "How many days do customers have to return an electronics item?", 14)
    add("policy", "How much does express shipping cost?", 9.99)
    add("policy", "What order value qualifies for free standard UK shipping?", 50)
    add("policy", "How long is the warranty on non-electronic products?", "1 year",
        lambda a: bool(re.search(r"\b(1|one)[- ]year", (a or "").lower())))
    add("policy", "How many loyalty points are needed for a £5 voucher?", 100)

    # --- multi-tool: chain several tools
    rev_de = _q(con, """SELECT SUM(o.quantity*p.price) FROM orders o JOIN products p ON p.id=o.product_id
                        JOIN customers c ON c.id=o.customer_id WHERE o.status='completed' AND c.country='Germany'""")
    add("multi_tool", "What is the total revenue of completed orders from customers in Germany, converted to EUR "
        "at the reference rate? Round to 2 decimals.", round(rev_de * 1.17, 2))
    avg_books = _q(con, "SELECT AVG(price) FROM products WHERE category='Books'")
    add("multi_tool", "What is the average price of Books products in USD at the reference rate? Round to 2 decimals.",
        round(avg_books * 1.27, 2))
    n_ref = _q(con, "SELECT COUNT(*) FROM orders WHERE status='refunded'")
    n_all = _q(con, "SELECT COUNT(*) FROM orders")
    add("multi_tool", "What percentage of all orders were refunded? Round to 2 decimals.", round(100 * n_ref / n_all, 2))
    big = _q(con, """SELECT MAX(o.quantity*p.price) FROM orders o JOIN products p ON p.id=o.product_id
                     WHERE o.status='completed'""")
    add("multi_tool", "How many loyalty points did the single largest completed order (by revenue) earn? "
        "Use the loyalty policy (points per whole £1, round down).", int(big))
    uk_total = _q(con, """SELECT SUM(o.quantity*p.price) FROM orders o JOIN products p ON p.id=o.product_id
                          JOIN customers c ON c.id=o.customer_id WHERE o.status='completed' AND c.country='UK'""")
    n_uk = _q(con, "SELECT COUNT(*) FROM customers WHERE country='UK'")
    add("multi_tool", "What is the average completed-order revenue per UK customer (total UK revenue divided by "
        "number of UK customers)? Round to 2 decimals.", round(uk_total / n_uk, 2))
    return T
