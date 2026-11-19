"""ЛР-2, вариант 9: мультитоварный поток минимальной стоимости (2 товара, общая ёмкость).

Запуск одной командой:
    python main.py
Результаты складываются в output/.
"""

import json
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.model import solve_lp
from src.verify import verify_solution, max_violation
from src.scenarios import run_all_scenarios
from src.milp import solve_milp
from src.visualize import plot_network

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "network.json")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def load_data():
    with open(DATA_PATH, encoding="utf-8") as f:
        return json.load(f)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    lines = []

    def log(msg):
        print(msg)
        lines.append(str(msg))

    data = load_data()
    log(f"Данные: {DATA_PATH}, seed={data['seed']}")
    for k, c in data["commodities"].items():
        log(f"  {k}: {c['source']} -> {c['sink']}, спрос {c['demand']}")

    # --- Базовое решение ---
    t0 = time.perf_counter()
    base_sol = solve_lp(data)
    solve_time = time.perf_counter() - t0
    solver_lines = [
        "Solver: HiGHS via scipy.optimize.linprog (method='highs')",
        f"Model: {len(base_sol['var_index'])} variables, "
        f"{len(data['commodities']) * len(data['nodes'])} equality rows, "
        f"{len(data['arcs'])} capacity rows",
        f"Base: status={base_sol['status_text']}, iterations={base_sol['nit']}, "
        f"time={solve_time:.3f}s",
        f"Message: {base_sol['message']}",
        "",
    ]
    if base_sol["status"] != 0:
        log(f"\nБазовый сценарий: status={base_sol['status_text']}: {base_sol['message']}")
        log("Базовая задача не решена — проверка невозможна.")
        return
    log(f"\nБазовый сценарий: status={base_sol['status_text']}, "
        f"min cost = {base_sol['objective']:.2f}, время {solve_time:.3f} c")

    rep = verify_solution(data, base_sol["flows"])
    log(f"Независимая проверка: ok={rep['ok']}, "
        f"max_violation={max_violation(rep):.2e}, "
        f"пересчитанная цель={rep['objective_recomputed']:.2f} "
        f"(расхождение с солвером {abs(rep['objective_recomputed'] - base_sol['objective']):.2e})")

    # Таблица решения: потоки по дугам
    rows = []
    for (k, a_id), v in base_sol["flows"].items():
        a = next(x for x in data["arcs"] if x["id"] == a_id)
        rows.append({"commodity": k, "arc": a_id, "flow": v,
                     "capacity_total": a["capacity"], "cost": a["cost"],
                     "cost_part": v * a["cost"]})
    sol_df = pd.DataFrame(rows).sort_values(["commodity", "arc"])
    sol_df.to_csv(os.path.join(OUT_DIR, "solution_base.csv"), index=False)
    log(f"Таблица потоков: {len(sol_df)} ненулевых переменных -> output/solution_base.csv")

    # Узкие места: насыщенные дуги и их теневые цены
    sat = [c for c in rep["checks"] if c["type"] == "capacity" and c["saturated"]]
    log(f"\nУзкие места (насыщенные дуги): {len(sat)}")
    bottlenecks = []
    for c in sat:
        shadow = -base_sol["marginals"]["arc_shadow"][c["arc"]]
        bottlenecks.append({"arc": c["arc"], "load": c["load"],
                            "capacity": c["capacity"], "shadow_price": shadow})
        log(f"  {c['arc']}: {c['load']:.1f}/{c['capacity']}, теневая цена = {shadow:.4f}")
    pd.DataFrame(bottlenecks).to_csv(os.path.join(OUT_DIR, "bottlenecks.csv"), index=False)

    plot_network(data, base_sol["flows"],
                 "Базовый сценарий: потоки (насыщенные дуги — оранжевые)",
                 "network_base.png")
    log("Схема сети: output/network_base.png")

    # --- Сценарный анализ ---
    log("\n=== Сценарный анализ ===")
    results = run_all_scenarios(data, log=log)
    sc_rows = []
    for sc in results:
        row = {"scenario": sc["name"]}
        if "sol" not in sc:
            row.update(status="feasible" if sc["feasible"] else "INFEASIBLE",
                       objective=sc["objective"], gap_vs_optimum_pct=sc["gap_pct"])
            if sc.get("overloaded"):
                row["overloaded_arcs"] = "; ".join(
                    f"{a}: {l:.1f}/{c}" for a, l, c in sc["overloaded"])
        else:
            row.update(status=sc["sol"]["status_text"], objective=sc["sol"]["objective"])
            if sc["sol"]["status"] == 0:
                row["delta_cost_vs_base"] = sc["sol"]["objective"] - base_sol["objective"]
                if sc["name"].startswith("arc_failure"):
                    plot_network(sc["data"], sc["sol"]["flows"],
                                 f"Сценарий отказа дуги {sc['failed_arc']}",
                                 "network_arc_failure.png", failed_arc=sc["failed_arc"])
        sc_rows.append(row)
        log("")
    pd.DataFrame(sc_rows).to_csv(os.path.join(OUT_DIR, "scenarios.csv"), index=False)
    log("Таблица сценариев: output/scenarios.csv")

    for sc in results:
        if "sol" in sc:
            solver_lines.append(
                f"{sc['name']}: status={sc['sol']['status_text']}, "
                f"iterations={sc['sol']['nit']}, objective={sc['sol']['objective']}")

    # --- Дополнительное задание: LP vs MILP ---
    log("\n=== Дополнительное задание: целочисленные рейсы (MILP) ===")
    t0 = time.perf_counter()
    lp_again = solve_lp(data)  # время LP для сравнения
    lp_time = time.perf_counter() - t0
    milp_sol = solve_milp(data)
    log(f"LP : status={lp_again['status_text']}, objective={lp_again['objective']:.2f}, время {lp_time:.3f} c")
    log(f"MILP: status={milp_sol['status_text']}, objective={milp_sol['objective']:.2f}, "
        f"время {milp_sol['time_s']:.3f} c, mip_gap={milp_sol['mip_gap']}")
    if milp_sol["status"] == 0:
        integrality_ok = all(abs(v - round(v)) < 1e-6
                             for v in milp_sol["flows"].values())
        rep_milp = verify_solution(data, milp_sol["flows"])
        gap = (milp_sol["objective"] - base_sol["objective"]) / base_sol["objective"] * 100
        log(f"Все потоки целочисленны: {integrality_ok}; "
            f"проверка допустимости: ok={rep_milp['ok']}, max_violation={max_violation(rep_milp):.2e}")
        log(f"Цена дискретности: +{milp_sol['objective'] - base_sol['objective']:.2f} "
            f"({gap:.2f}%) к LP-оптимуму")
        milp_rows = [{"commodity": k, "arc": a, "flow": v}
                     for (k, a), v in sorted(milp_sol["flows"].items())]
        pd.DataFrame(milp_rows).to_csv(os.path.join(OUT_DIR, "solution_milp.csv"), index=False)
        pd.DataFrame([
            {"version": "LP", "objective": base_sol["objective"], "time_s": lp_time},
            {"version": "MILP", "objective": milp_sol["objective"],
             "time_s": milp_sol["time_s"], "mip_gap": milp_sol["mip_gap"],
             "cost_of_discrete_pct": gap},
        ]).to_csv(os.path.join(OUT_DIR, "lp_vs_milp.csv"), index=False)

    solver_lines += [
        "",
        f"MILP (scipy.optimize.milp, HiGHS): status={milp_sol['status_text']}, "
        f"objective={milp_sol['objective']}, time={milp_sol['time_s']:.3f}s, "
        f"mip_gap={milp_sol['mip_gap']}",
    ]
    with open(os.path.join(OUT_DIR, "solver_log.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(solver_lines) + "\n")

    with open(os.path.join(OUT_DIR, "run_log.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log("\nВсе результаты в output/")


if __name__ == "__main__":
    main()
