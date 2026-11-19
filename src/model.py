"""Построение и решение LP-модели мультитоварного потока минимальной стоимости.

Модель:
  Переменные: x[k, a] >= 0 — поток товара k по дуге a (в единицах груза).
  Цель:      min sum_{k,a} cost[a] * x[k, a]            (стоимость перевозок, у.е.)
  Ограничения баланса в каждой вершине v для каждого товара k:
      сумма исходящих x[k, .] - сумма входящих x[k, .] =
          +demand[k]  при v = source(k)
          -demand[k]  при v = sink(k)
          0           иначе
  Общая пропускная способность дуг:
      sum_k x[k, a] <= capacity[a]   для каждой дуги a.
"""

import numpy as np
from scipy.optimize import linprog

TOL = 1e-9


def build_problem(data):
    """Возвращает (commodities, arcs, A_eq, b_eq, A_ub, b_ub, c, var_index)."""
    nodes = data["nodes"]
    arcs = data["arcs"]
    comms = list(data["commodities"].keys())

    var_index = {}
    n = 0
    for k in comms:
        for a in arcs:
            var_index[(k, a["id"])] = n
            n += 1

    # Балансовые ограничения: len(comms) * len(nodes) равенств
    n_eq = len(comms) * len(nodes)
    A_eq = np.zeros((n_eq, n))
    b_eq = np.zeros(n_eq)
    for i, k in enumerate(comms):
        info = data["commodities"][k]
        for j, v in enumerate(nodes):
            row = i * len(nodes) + j
            for a in arcs:
                if a["from"] == v:
                    A_eq[row, var_index[(k, a["id"])]] = 1.0
                if a["to"] == v:
                    A_eq[row, var_index[(k, a["id"])]] = -1.0
            if v == info["source"]:
                b_eq[row] = info["demand"]
            elif v == info["sink"]:
                b_eq[row] = -info["demand"]

    # Общая пропускная способность: одно неравенство на дугу
    n_ub = len(arcs)
    A_ub = np.zeros((n_ub, n))
    b_ub = np.zeros(n_ub)
    for j, a in enumerate(arcs):
        for k in comms:
            A_ub[j, var_index[(k, a["id"])]] = 1.0
        b_ub[j] = a["capacity"]

    c = np.zeros(n)
    for k in comms:
        for a in arcs:
            c[var_index[(k, a["id"])]] = a["cost"]

    return comms, arcs, A_eq, b_eq, A_ub, b_ub, c, var_index


def solve_lp(data, method="highs"):
    """Решает LP солвером HiGHS (scipy.optimize.linprog)."""
    comms, arcs, A_eq, b_eq, A_ub, b_ub, c, var_index = build_problem(data)
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=(0, None), method=method,
                  options={"disp": True})
    return {
        "status": res.status,
        "status_text": _status_text(res.status),
        "message": res.message,
        "objective": float(res.fun) if res.status == 0 else None,
        "nit": res.nit,
        "x": res.x if res.status == 0 else None,
        "flows": _flows_from_x(res.x, var_index) if res.status == 0 else None,
        "marginals": {  # теневые цены: равенства -> потенциалы, неравенств -> дефициты дуг
            "node_potentials": dict(zip(
                [(k, v) for k in comms for v in data["nodes"]],
                res.eqlin.marginals if res.status == 0 else [])),
            "arc_shadow": dict(zip(
                [a["id"] for a in arcs],
                res.ineqlin.marginals if res.status == 0 else [])),
        } if res.status == 0 else None,
        "var_index": var_index,
    }


def _status_text(status):
    return {0: "optimal", 1: "iteration limit", 2: "infeasible",
            3: "unbounded", 4: "numerical issue"}.get(status, f"unknown ({status})")


def _flows_from_x(x, var_index):
    return {key: float(x[idx]) for key, idx in var_index.items() if abs(x[idx]) > TOL}
