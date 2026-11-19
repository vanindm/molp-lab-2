"""Дополнительное задание: целочисленная версия (неделимые рейсы) и сравнение LP/MILP.

Поток каждого товара по каждой дуге становится целочисленным
(integrality=1, т.е. переменные из Z+). Физический смысл: груз отгружается
неделимыми партиями (рейсами) по 1 единице; маршрут распадается на
целочисленные потоки по дугам (партии могут идти разными путями, но каждая
партия целая).

Сравниваются: целевая функция, время решения, структура потоков.
"""

import time

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

from .model import build_problem


def solve_milp(data):
    comms, arcs, A_eq, b_eq, A_ub, b_ub, c, var_index = build_problem(data)
    n = len(c)
    t0 = time.perf_counter()
    res = milp(
        c,
        constraints=[
            LinearConstraint(A_eq, b_eq, b_eq),
            LinearConstraint(A_ub, -np.inf, b_ub),
        ],
        bounds=Bounds(np.zeros(n), np.full(n, np.inf)),
        integrality=np.ones(n),
    )
    elapsed = time.perf_counter() - t0
    flows = {}
    if res.status == 0:
        for key, idx in var_index.items():
            if abs(res.x[idx]) > 1e-9:
                flows[key] = float(res.x[idx])
    return {
        "status": res.status,
        "status_text": {0: "optimal", 1: "iteration limit", 2: "infeasible",
                        3: "unbounded", 4: "numerical issue"}.get(res.status, str(res.status)),
        "objective": float(res.fun) if res.status == 0 else None,
        "flows": flows,
        "time_s": elapsed,
        "mip_gap": getattr(res, "mip_gap", None),
        "var_index": var_index,
    }
