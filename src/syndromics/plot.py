"""Chip map: draw diagnostics findings on the qubit lattice."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle

from .diagnose import Report, qubit_roles

RED, AMB, OK, LINE = "#c8102e", "#f2a900", "#e6e8ec", "#4a4f57"


def _draw_lattice(ax, coords, data, meas, title):
    ax.set_title(title, fontsize=12, pad=8)
    for q in data:
        x, y = coords[q]
        ax.add_patch(Circle((x, y), 0.32, fc=OK, ec=LINE, lw=0.8, zorder=2))
    for q in meas:
        x, y = coords[q]
        ax.add_patch(Rectangle((x - .25, y - .25), .5, .5, fc="white", ec="#9aa0a8", lw=0.8, zorder=2))
    xs = [c[0] for c in coords.values()]; ys = [c[1] for c in coords.values()]
    ax.set_xlim(min(xs) - 1, max(xs) + 1); ax.set_ylim(max(ys) + 1, min(ys) - 1)
    ax.set_aspect("equal"); ax.axis("off")


def _mark(ax, coords, kind, qs, color, label=None, dashed=False):
    ls = (0, (3, 2)) if dashed else "-"
    if kind == "data_qubit":
        x, y = coords[qs[0]]
        ax.add_patch(Circle((x, y), 0.32, fc=color, ec="k", lw=1.2, ls=ls, zorder=3))
    elif kind == "measure_qubit":
        x, y = coords[qs[0]]
        ax.add_patch(Rectangle((x - .25, y - .25), .5, .5, fc=color, ec="k", lw=1.2, ls=ls, zorder=3))
    else:
        (x1, y1), (x2, y2) = coords[qs[0]], coords[qs[1]]
        ax.plot([x1, x2], [y1, y2], color=color, lw=5, ls=ls, solid_capstyle="round", zorder=1.5)
        x, y = (x1 + x2) / 2, (y1 + y2) / 2
    if label:
        ax.text(x, y - 0.55, label, ha="center", fontsize=8, zorder=4)


def plot_report(report: Report, nominal, path: str, truth: dict | None = None, title: str = ""):
    coords, data, meas = qubit_roles(nominal.flattened())
    ncols = 2 if truth else 1
    fig, axes = plt.subplots(1, ncols, figsize=(6.5 * ncols, 6.6))
    axes = [axes] if ncols == 1 else list(axes)
    if truth:
        _draw_lattice(axes[0], coords, data, meas, "Hidden truth (simulator only)")
        for q in truth["bad_data_qubits"]:
            _mark(axes[0], coords, "data_qubit", [q], RED)
        for a, b in truth["crosstalk_pairs"]:
            _mark(axes[0], coords, "crosstalk", [a, b], RED)
        for q in truth["bad_readout_qubits"]:
            _mark(axes[0], coords, "measure_qubit", [q], RED)
    ax = axes[-1]
    _draw_lattice(ax, coords, data, meas, "Diagnosed from syndrome data")
    exact, amb = {}, set()
    for f in report.flagged:
        keys = {(k, q) for k, q, _ in f.members}
        if f.exact:
            k = next(iter(keys)); exact[k] = max(exact.get(k, 0), f.times_spec)
        else:
            amb |= keys
    for k, q in amb - set(exact):
        _mark(ax, coords, k, list(q), AMB, dashed=True)
    for (k, q), xs in exact.items():
        _mark(ax, coords, k, list(q), RED, label=f"{xs:.1f}x")
    handles = [Line2D([], [], marker="o", ls="", mfc=OK, mec=LINE, ms=11, label="data qubit"),
               Line2D([], [], marker="s", ls="", mfc="white", mec="#9aa0a8", ms=10, label="measure qubit"),
               Line2D([], [], marker="o", ls="", mfc=RED, mec="k", ms=11, label="defect, x times worse than spec"),
               Line2D([], [], marker="o", ls="", mfc=AMB, mec="k", ms=11, label="one of a syndrome-equivalent group"),
               Line2D([], [], color=RED, lw=5, label="crosstalk pair")]
    fig.legend(handles=handles, loc="lower center", ncol=3 if ncols == 1 else 5, frameon=False, fontsize=9)
    if title:
        fig.suptitle(title, fontsize=13, weight="bold")
    plt.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------- device-wide placement map
def device_coords(backend):
    """2-D positions for a backend's qubits. IBM heavy-hex (Heron r2, 156 qubits): rows of 16
    joined by bridge qubits; anything else falls back to a spring layout."""
    edges = {tuple(sorted(e)) for e in backend.coupling_map.get_edges()}
    n = backend.num_qubits
    nbr = {q: set() for q in range(n)}
    for a, b in edges:
        nbr[a].add(b); nbr[b].add(a)
    if n == 156:
        pos = {}
        for r in range(8):
            for i in range(16):
                pos[20 * r + i] = (i, 2 * r)
        for r in range(7):
            for j in range(4):
                q = 20 * r + 16 + j
                up = min(nbr[q])
                pos[q] = (pos[up][0], 2 * r + 1)
        if len(pos) == n:
            return pos, edges
    import rustworkx as rx
    g = rx.PyGraph(); g.add_nodes_from(range(n)); g.add_edges_from_no_data(list(edges))
    lay = rx.spring_layout(g, seed=1)
    return {q: (lay[q][0] * 12, lay[q][1] * 12) for q in range(n)}, edges


def plot_placement(backend, health, panels, path, suptitle=None):
    """panels: [(title, used_qubits, used_pairs)]. Draws today's problems in red, problems the
    calibration already knows in dark grey, and each placement in colour."""
    pos, edges = device_coords(backend)
    bad_q = set(health.bad_qubits()) if health else set()
    bad_p = set(health.bad_pairs()) if health else set()
    known_q = {int(k[1:]) for k in (health.known_bad if health else {}) if "-" not in k}
    fig, axes = plt.subplots(1, len(panels), figsize=(7.2 * len(panels), 6.4))
    axes = [axes] if len(panels) == 1 else axes
    for ax, (title, used, pairs, colour) in zip(axes, panels):
        used, pairs = set(used), {tuple(sorted(p)) for p in pairs}
        for a, b in edges:
            (x1, y1), (x2, y2) = pos[a], pos[b]
            if (a, b) in bad_p:
                ax.plot([x1, x2], [y1, y2], color=RED, lw=3.2, zorder=1)
            elif (a, b) in pairs:
                ax.plot([x1, x2], [y1, y2], color=colour, lw=3.0, zorder=1)
            else:
                ax.plot([x1, x2], [y1, y2], color="#c9cdd3", lw=0.9, zorder=0)
        for q, (x, y) in pos.items():
            fc, ec, lw = OK, "#9aa0a8", 0.6
            if q in known_q:
                fc = "#6b7078"
            if q in used:
                fc, ec, lw = colour, "black", 0.9
            if q in bad_q:
                ec, lw = RED, 2.6
            ax.add_patch(Circle((x, y), 0.33, fc=fc, ec=ec, lw=lw, zorder=2))
        xs = [p[0] for p in pos.values()]; ys = [p[1] for p in pos.values()]
        ax.set_xlim(min(xs) - 1, max(xs) + 1); ax.set_ylim(max(ys) + 1, min(ys) - 1)
        ax.set_aspect("equal"); ax.axis("off"); ax.set_title(title, fontsize=12.5, pad=6)
    handles = [Line2D([], [], color=RED, lw=3, label="changed since calibration (found by Syndromics)"),
               Line2D([], [], marker="o", ls="", mfc="#6b7078", mec="#6b7078", ms=9,
                      label="bad in published calibration"),
               Line2D([], [], marker="o", ls="", mfc=panels[0][3], mec="black", ms=9, label=panels[0][0].split("\n")[0]),
               ]
    if len(panels) > 1:
        handles.append(Line2D([], [], marker="o", ls="", mfc=panels[1][3], mec="black", ms=9,
                              label=panels[1][0].split("\n")[0]))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=10.5)
    if suptitle:
        fig.suptitle(suptitle, fontsize=14, y=0.99)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(path, dpi=130)
    plt.close(fig)
