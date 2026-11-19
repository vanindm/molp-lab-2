"""Визуализация сети с пропускными способностями и найденными потоками (NetworkX)."""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")


def plot_network(data, flows, title, filename, failed_arc=None):
    G = nx.DiGraph()
    for n in data["nodes"]:
        G.add_node(n)
    for a in data["arcs"]:
        G.add_edge(a["from"], a["to"], id=a["id"])

    pos = {n: tuple(p) for n, p in data["pos"].items()}

    # Суммарная загрузка дуги обоими товарами
    load = {}
    for (k, a_id), v in (flows or {}).items():
        load[a_id] = load.get(a_id, 0.0) + v

    fig, ax = plt.subplots(figsize=(11, 8))
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color="#a8d8ff",
                           edgecolors="#2266aa", ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold", ax=ax)

    for a in data["arcs"]:
        u, v = a["from"], a["to"]
        w = load.get(a["id"], 0.0)
        color = "#cc2222" if a["id"] == failed_arc else (
            "#ee8800" if w >= a["capacity"] - 1e-7 else "#4477aa")
        width = 1.0 + 4.0 * (w / a["capacity"] if a["capacity"] else 0)
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], edge_color=color,
                               width=width, arrows=True, arrowsize=18,
                               connectionstyle="arc3,rad=0.08", ax=ax)
        mx, my = (pos[u][0] + pos[v][0]) / 2, (pos[u][1] + pos[v][1]) / 2
        label = f"{a['id']}\n{w:.1f}/{a['capacity']}"
        if w > 0:
            label += f" (c={a['cost']})"
        ax.text(mx, my, label, fontsize=7, ha="center",
                color="#cc2222" if a["id"] == failed_arc else "#333333",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.8, lw=0))

    for k, info in data["commodities"].items():
        ax.text(pos[info["source"]][0], pos[info["source"]][1] + 0.55,
                f"{k}: +{info['demand']}", fontsize=9, ha="center", color="#117722")
        ax.text(pos[info["sink"]][0], pos[info["sink"]][1] - 0.75,
                f"{k}: −{info['demand']}", fontsize=9, ha="center", color="#aa1111")

    ax.set_title(title)
    ax.axis("off")
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, filename)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path
