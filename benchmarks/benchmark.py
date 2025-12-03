#!/usr/bin/env python3
"""Benchmarks comparing DequeDict to dict, deque, and OrderedDict.

Usage:
    python benchmarks/benchmark.py          # Uses C extension
    NOC=1 python benchmarks/benchmark.py    # Uses pure Python
"""
from __future__ import annotations

import time
from collections import OrderedDict, deque
from typing import Any, Callable

from dequedict import DequeDict


def benchmark(func: Callable[[], Any], iterations: int = 100_000) -> float:
    """Run benchmark and return time per operation in nanoseconds."""
    start = time.perf_counter()
    for _ in range(iterations):
        func()
    elapsed = time.perf_counter() - start
    return (elapsed / iterations) * 1e9


def format_ns(ns: float) -> str:
    """Format nanoseconds as ns or µs."""
    if ns >= 1000:
        return f"{ns/1000:>7.2f} µs"
    return f"{ns:>7.1f} ns"


def main() -> None:
    print("=" * 72)
    print("DequeDict Benchmarks")
    print("=" * 72)

    is_c = "abc.ABCMeta" not in str(type(DequeDict))
    print(f"Implementation: {'C extension' if is_c else 'Pure Python'}\n")

    n = 1000
    keys = [f"key_{i}" for i in range(n)]
    values = list(range(n))

    dd = DequeDict(zip(keys, values))
    d = dict(zip(keys, values))
    od = OrderedDict(zip(keys, values))
    dq = deque(zip(keys, values))

    lookup_key = keys[n // 2]

    print("--- Key Lookup ---")
    print(f"  DequeDict   : {format_ns(benchmark(lambda: dd[lookup_key], 1_000_000))}")
    print(f"  dict        : {format_ns(benchmark(lambda: d[lookup_key], 1_000_000))}")
    print(f"  OrderedDict : {format_ns(benchmark(lambda: od[lookup_key], 1_000_000))}")

    print("\n--- Contains ---")
    print(f"  DequeDict   : {format_ns(benchmark(lambda: lookup_key in dd, 1_000_000))}")
    print(f"  dict        : {format_ns(benchmark(lambda: lookup_key in d, 1_000_000))}")
    print(f"  OrderedDict : {format_ns(benchmark(lambda: lookup_key in od, 1_000_000))}")
    print("  deque        : O(n)")

    print("\n--- Peek Left ---")
    print(f"  DequeDict   : {format_ns(benchmark(lambda: dd.peekleft(), 1_000_000))}")
    print(f"  OrderedDict : {format_ns(benchmark(lambda: next(iter(od.values())), 1_000_000))}")
    print(f"  deque       : {format_ns(benchmark(lambda: dq[0], 1_000_000))}")

    print("\n--- Peek Right ---")
    print(f"  DequeDict   : {format_ns(benchmark(lambda: dd.peek(), 1_000_000))}")
    print(f"  OrderedDict : {format_ns(benchmark(lambda: next(reversed(od.values())), 1_000_000))}")
    print(f"  deque       : {format_ns(benchmark(lambda: dq[-1], 1_000_000))}")

    dd_move = DequeDict(zip(keys, values))
    od_move = OrderedDict(zip(keys, values))

    print("\n--- Move to End ---")
    print(f"  DequeDict   : {format_ns(benchmark(lambda: dd_move.move_to_end(lookup_key), 1_000_000))}")
    print(f"  OrderedDict : {format_ns(benchmark(lambda: od_move.move_to_end(lookup_key), 1_000_000))}")

    def insert_dd():
        x = DequeDict()
        for i in range(100):
            x[f"k{i}"] = i

    def insert_d():
        x = {}
        for i in range(100):
            x[f"k{i}"] = i

    def insert_od():
        x = OrderedDict()
        for i in range(100):
            x[f"k{i}"] = i

    print("\n--- Insert 100 Items ---")
    print(f"  DequeDict   : {format_ns(benchmark(insert_dd, 10_000))}")
    print(f"  dict        : {format_ns(benchmark(insert_d, 10_000))}")
    print(f"  OrderedDict : {format_ns(benchmark(insert_od, 10_000))}")

    def pop_left_dd():
        x = DequeDict(zip(keys[:100], values[:100]))
        for _ in range(100):
            x.popleft()

    def pop_left_od():
        x = OrderedDict(zip(keys[:100], values[:100]))
        for _ in range(100):
            x.popitem(last=False)

    def pop_left_dq():
        x = deque(zip(keys[:100], values[:100]))
        for _ in range(100):
            x.popleft()

    print("\n--- Pop Left 100 Items ---")
    print(f"  DequeDict   : {format_ns(benchmark(pop_left_dd, 10_000))}")
    print(f"  OrderedDict : {format_ns(benchmark(pop_left_od, 10_000))}")
    print(f"  deque       : {format_ns(benchmark(pop_left_dq, 10_000))}")
    print("  dict         : N/A")

    def pop_right_dd():
        x = DequeDict(zip(keys[:100], values[:100]))
        for _ in range(100):
            x.pop()

    def pop_right_d():
        x = dict(zip(keys[:100], values[:100]))
        for _ in range(100):
            x.popitem()

    def pop_right_dq():
        x = deque(zip(keys[:100], values[:100]))
        for _ in range(100):
            x.pop()

    print("\n--- Pop Right 100 Items ---")
    print(f"  DequeDict   : {format_ns(benchmark(pop_right_dd, 10_000))}")
    print(f"  dict        : {format_ns(benchmark(pop_right_d, 10_000))}")
    print(f"  deque       : {format_ns(benchmark(pop_right_dq, 10_000))}")

    def del_dd():
        x = DequeDict(zip(keys[:100], values[:100]))
        for k in keys[:100]:
            del x[k]

    def del_d():
        x = dict(zip(keys[:100], values[:100]))
        for k in keys[:100]:
            del x[k]

    def del_od():
        x = OrderedDict(zip(keys[:100], values[:100]))
        for k in keys[:100]:
            del x[k]

    print("\n--- Delete 100 Items by Key ---")
    print(f"  DequeDict   : {format_ns(benchmark(del_dd, 10_000))}")
    print(f"  dict        : {format_ns(benchmark(del_d, 10_000))}")
    print(f"  OrderedDict : {format_ns(benchmark(del_od, 10_000))}")
    print("  deque        : O(n)")

    print("\n--- Iterate 1000 Items ---")
    print(f"  DequeDict   : {format_ns(benchmark(lambda: list(dd.items()), 10_000))}")
    print(f"  dict        : {format_ns(benchmark(lambda: list(d.items()), 10_000))}")
    print(f"  OrderedDict : {format_ns(benchmark(lambda: list(od.items()), 10_000))}")
    print(f"  deque       : {format_ns(benchmark(lambda: list(dq), 10_000))}")

    print("\n" + "=" * 72)


if __name__ == "__main__":
    main()
