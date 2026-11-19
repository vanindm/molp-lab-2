"""Сценарный анализ: базовый, отказ дуги, рост спроса, стрессовый, жадная эвристика."""

import copy

from .model import solve_lp
from .heuristic import greedy_residual, greedy_naive
from .verify import verify_solution, max_violation


def scenario_base(data):
    sol = solve_lp(data)
    return {"name": "base", "data": data, "sol": sol}


def scenario_arc_failure(data, base_sol):
    """Закрыть самую загруженную дугу базового решения и перерешать."""
    arc_loads = {}
    for (k, a_id), v in base_sol["flows"].items():
        arc_loads[a_id] = arc_loads.get(a_id, 0.0) + v
    worst = max(arc_loads, key=arc_loads.get)

    d = copy.deepcopy(data)
    d["arcs"] = [a for a in d["arcs"] if a["id"] != worst]
    sol = solve_lp(d)
    return {"name": f"arc_failure ({worst})", "data": d, "sol": sol,
            "failed_arc": worst, "lost_load": arc_loads[worst]}


def scenario_demand_growth(data, factor=1.2):
    """Рост спроса на factor для всех товаров."""
    d = copy.deepcopy(data)
    for k in d["commodities"]:
        d["commodities"][k]["demand"] = round(d["commodities"][k]["demand"] * factor)
    sol = solve_lp(d)
    return {"name": f"demand_x{factor}", "data": d, "sol": sol}


def scenario_stress(data, factor=1.4, cap_cut=0.75):
    """Стрессовый: рост спроса на factor И урезание пропускных способностей до cap_cut."""
    d = copy.deepcopy(data)
    for k in d["commodities"]:
        d["commodities"][k]["demand"] = round(d["commodities"][k]["demand"] * factor)
    for a in d["arcs"]:
        a["capacity"] = max(1, int(a["capacity"] * cap_cut))
    sol = solve_lp(d)
    return {"name": f"stress (demand_x{factor}, cap_x{cap_cut})", "data": d, "sol": sol}


def run_all_scenarios(data, log=print):
    results = []

    base = scenario_base(data)
    results.append(base)
    _log_sol(log, base)

    fail = scenario_arc_failure(data, base["sol"])
    results.append(fail)
    _log_sol(log, fail)

    growth = scenario_demand_growth(data)
    results.append(growth)
    _log_sol(log, growth)

    stress = scenario_stress(data)
    results.append(stress)
    _log_sol(log, stress)

    # Жадные эвристики на каждом сценарии с допустимым LP-решением
    for sc in list(results):
        if sc["sol"]["status"] != 0:
            continue
        flows, feasible, obj = greedy_residual(sc["data"])
        gap = (obj - sc["sol"]["objective"]) / sc["sol"]["objective"] * 100
        results.append({"name": f"greedy_residual @ {sc['name']}", "flows": flows,
                        "feasible": feasible, "objective": obj, "gap_pct": gap})
        log(f"[greedy_residual @ {sc['name']}] feasible={feasible}, "
            f"objective={obj:.2f}, gap={round(gap, 2)}%")

        flows, feasible, obj, overloaded = greedy_naive(sc["data"])
        gap = (obj - sc["sol"]["objective"]) / sc["sol"]["objective"] * 100
        results.append({"name": f"greedy_naive @ {sc['name']}", "flows": flows,
                        "feasible": feasible, "objective": obj, "gap_pct": gap,
                        "overloaded": overloaded})
        log(f"[greedy_naive @ {sc['name']}] feasible={feasible}, "
            f"objective={obj:.2f}, gap={round(gap, 2)}%, "
            f"перегруженных дуг: {len(overloaded)}")

    return results


def _log_sol(log, sc):
    sol = sc["sol"]
    if sol["status"] == 0:
        rep = verify_solution(sc["data"], sol["flows"])
        log(f"[{sc['name']}] status={sol['status_text']}, "
            f"objective={sol['objective']:.2f}, max_violation={max_violation(rep):.2e}")
    else:
        log(f"[{sc['name']}] status={sol['status_text']}: {sol['message']}")
