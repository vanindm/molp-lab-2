"""Жадные эвристики для мультитоварного потока.

1. greedy_residual — «сначала самые дешёвые маршруты» с учётом остаточных
   пропускных способностей: Dijkstra по стоимости, отправка максимума,
   повтор. Не даёт гарантий оптимальности, но учитывает общие дуги.
2. greedy_naive — ещё проще: каждый товар целиком по своему кратчайшему
   пути по стоимости, ёмкости игнорируются. Показывает риск «в лоб»
   планирования: перегрузка дуг.
"""

import heapq


def greedy_residual(data):
    """Возвращает (flows, feasible, objective)."""
    residual = {a["id"]: a["capacity"] for a in data["arcs"]}
    adj = {}
    for a in data["arcs"]:
        adj.setdefault(a["from"], []).append(a)

    flows = {}
    feasible = True
    for k, info in data["commodities"].items():
        remaining = info["demand"]
        while remaining > 1e-9:
            path = _cheapest_path(adj, info["source"], info["sink"], residual)
            if path is None:
                feasible = False
                break
            bottleneck = min(residual[a["id"]] for a in path)
            send = min(bottleneck, remaining)
            for a in path:
                residual[a["id"]] -= send
                flows[(k, a["id"])] = flows.get((k, a["id"]), 0.0) + send
            remaining -= send
        if not feasible:
            break

    return flows, feasible, _objective(data, flows)


def greedy_naive(data):
    """Каждый товар целиком по кратчайшему пути по стоимости, без учёта ёмкостей.

    Возвращает (flows, feasible, objective, overloaded)."""
    residual = {a["id"]: float("inf") for a in data["arcs"]}
    adj = {}
    for a in data["arcs"]:
        adj.setdefault(a["from"], []).append(a)

    flows = {}
    reachable = True
    for k, info in data["commodities"].items():
        path = _cheapest_path(adj, info["source"], info["sink"], residual)
        if path is None:
            reachable = False
            break
        for a in path:
            flows[(k, a["id"])] = flows.get((k, a["id"]), 0.0) + info["demand"]

    overloaded = []
    for a in data["arcs"]:
        load = sum(v for (kk, a_id), v in flows.items() if a_id == a["id"])
        if load > a["capacity"] + 1e-9:
            overloaded.append((a["id"], load, a["capacity"]))
    return flows, reachable and not overloaded, _objective(data, flows), overloaded


def _objective(data, flows):
    cost_by_arc = {a["id"]: a["cost"] for a in data["arcs"]}
    return sum(cost_by_arc[a_id] * v for (k, a_id), v in flows.items())


def _cheapest_path(adj, s, t, residual):
    """Dijkstra по дугам с остаточной пропускной способностью > 0. Возвращает список дуг."""
    dist = {s: 0.0}
    prev = {}
    pq = [(0.0, s)]
    while pq:
        d, u = heapq.heappop(pq)
        if u == t:
            break
        if d > dist.get(u, float("inf")):
            continue
        for a in adj.get(u, []):
            if residual[a["id"]] <= 1e-9:
                continue
            nd = d + a["cost"]
            if nd < dist.get(a["to"], float("inf")):
                dist[a["to"]] = nd
                prev[a["to"]] = a
                heapq.heappush(pq, (nd, a["to"]))
    if t not in prev and t != s:
        return None
    path = []
    u = t
    while u != s:
        a = prev[u]
        path.append(a)
        u = a["from"]
    return path[::-1]
