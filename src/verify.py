"""Независимая проверка решения кодом на NumPy без обращения к объекту солвера.

Проверяет:
  1. Баланс потока каждого товара в каждой вершине (допуск 1e-7);
  2. Неотрицательность переменных и границы общей пропускной способности;
  3. Пересчёт значения целевой функции по данным (не по ответу солвера).

Все вычисления векторизованы: баланс строится матрицей инцидентности,
умноженной на вектор потоков; объект солвера нигде не используется.
"""

import numpy as np

TOL = 1e-7


def verify_solution(data, flows):
    """flows: dict {(commodity, arc_id): value}. Возвращает отчёт проверки."""
    nodes = data["nodes"]
    comms = list(data["commodities"])
    arcs = data["arcs"]
    arc_ids = [a["id"] for a in arcs]
    cap = np.array([a["capacity"] for a in arcs])
    cost = np.array([a["cost"] for a in arcs])

    keys = [(k, a_id) for k in comms for a_id in arc_ids]
    x = np.array([flows.get(key, 0.0) for key in keys])

    report = {"checks": [], "ok": True}

    def add(check):
        report["checks"].append(check)
        report["ok"] = report["ok"] and check["ok"]

    # --- 1. Баланс: B_k @ x_k = b_k для каждого товара ---
    # B — матрица инцидентности (nodes x arcs): +1 исходящая, -1 входящая
    node_idx = {v: j for j, v in enumerate(nodes)}
    B = np.zeros((len(nodes), len(arcs)))
    for r, a in enumerate(arcs):
        B[node_idx[a["from"]], r] = 1.0
        B[node_idx[a["to"]], r] = -1.0

    for i, k in enumerate(comms):
        info = data["commodities"][k]
        xk = x[i * len(arcs):(i + 1) * len(arcs)]
        b = np.zeros(len(nodes))
        b[node_idx[info["source"]]] = info["demand"]
        b[node_idx[info["sink"]]] = -info["demand"]
        viol = np.abs(B @ xk - b)
        for v, val in zip(nodes, viol):
            add({"type": "balance", "commodity": k, "node": v,
                 "violation": float(val), "ok": val <= TOL})

    # --- 2. Неотрицательность и общая пропускная способность ---
    neg = x < -TOL
    for m in np.nonzero(neg)[0]:
        add({"type": "nonneg", "commodity": keys[m][0], "arc": keys[m][1],
             "value": float(x[m]), "ok": False})

    loads = np.zeros(len(arcs))
    for i in range(len(comms)):
        loads += x[i * len(arcs):(i + 1) * len(arcs)]
    viol = np.maximum(0.0, loads - cap)
    slack = cap - loads
    for r, a in enumerate(arcs):
        add({"type": "capacity", "arc": a["id"], "load": float(loads[r]),
             "capacity": float(cap[r]), "slack": float(slack[r]),
             "violation": float(viol[r]), "saturated": slack[r] <= TOL,
             "ok": viol[r] <= TOL})

    # --- 3. Пересчёт целевой функции по данным ---
    total = 0.0
    for i in range(len(comms)):
        xk = x[i * len(arcs):(i + 1) * len(arcs)]
        total += float(cost @ xk)
    report["objective_recomputed"] = total
    return report


def max_violation(report):
    return max(c.get("violation", 0.0) for c in report["checks"])
