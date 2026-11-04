"""Figures for the benchmark matrix, drawn from `observations.jsonl` only.

Every figure is written as SVG, with text kept as text for the docs site.
Colours follow the simulator, never its rank, so a simulator keeps one hue on
every figure. Missing data is drawn as a hatched cell or an open marker with
its status (not applicable, timeout, crash, invalid), never as zero.
"""

from __future__ import annotations

import math
import os
from collections import defaultdict
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

from . import aggregate, stats  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
NEUTRAL = "#f0efec"
DE_EMPHASIS = "#b9b8b1"

COLORS = {
    "prismq": "#2a78d6",
    "aer-statevector": "#eb6834",
    "aer-stabilizer": "#eb6834",
    "aer-mps": "#eb6834",
    "qsim": "#1baf7a",
    "quest": "#eda100",
    "spinoza": "#e87ba4",
    "qip": "#008300",
    "aer-automatic": "#6250d6",
    "aer-statevector-f32": "#e34948",
    # Stim only runs the Clifford rows, where qsim never appears; fig2 puts those rows in their own panel.
    "stim": "#1baf7a",
}
SHORT = {
    "prismq": "PRISM-Q",
    "aer-statevector": "Aer statevector",
    "aer-statevector-f32": "Aer statevector (fp32)",
    "aer-automatic": "Aer automatic",
    "aer-stabilizer": "Aer stabilizer",
    "aer-mps": "Aer MPS",
    "qsim": "qsim (fp32)",
    "quest": "QuEST",
    "spinoza": "Spinoza",
    "qip": "qip",
    "stim": "Stim",
}
DIVERGING = LinearSegmentedColormap.from_list(
    "prismq_div", ["#b8322f", "#e34948", "#f2a3a0", NEUTRAL, "#9ec5f4", "#3987e5", "#184f95"]
)


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
            "font.size": 9,
            "svg.fonttype": "none",
            "svg.hashsalt": "prism-q-matrix",
            "axes.facecolor": SURFACE,
            "figure.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 10,
            "axes.titleweight": "semibold",
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_2,
            "ytick.labelcolor": INK_2,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "lines.linewidth": 2.0,
            "lines.markersize": 5,
        }
    )


def _color(name: str) -> str:
    return COLORS.get(name.partition(":")[0], MUTED)


def _label(name: str) -> str:
    base, _, option = name.partition(":")
    return SHORT.get(base, base) + (f" ({option})" if option else "")


def _save(fig, out_dir: str, stem: str, written: list[str]) -> None:
    path = os.path.join(out_dir, f"{stem}.svg")
    fig.savefig(path, bbox_inches="tight", dpi=200, metadata={"Date": None})
    written.append(path)
    plt.close(fig)


def _row_label(wl: dict) -> str:
    p = wl["params"]
    bits = [wl["family"], f"{wl['num_qubits']}q"]
    if "depth" in p and wl["family"] in ("rcs", "lowent"):
        bits.append(f"d{p['depth']}")
    if wl["family"] == "disjoint":
        bits.append(p.get("layout", ""))
    if wl["mode"] == "counts":
        bits.append(_shots(wl["shots"]))
    return " ".join(bits)


def _shots(n: int) -> str:
    if n >= 1_000_000:
        return f"{n // 1_000_000}M shots"
    if n >= 1000:
        return f"{n // 1000}k shots"
    return f"{n} shot" + ("s" if n != 1 else "")


def _status(record: dict) -> str:
    if record["status"] != "ok":
        return {"not-applicable": "n/a"}.get(record["status"], record["status"])
    v = record.get("validation") or {}
    if v.get("pass") is False:
        return "invalid"
    return "ok"


def _family_order(wl: dict) -> tuple:
    order = ["rcs", "qft", "qv", "clifford", "lowent", "disjoint", "midcircuit", "sampling"]
    fam = wl["family"]
    return (
        order.index(fam) if fam in order else 99,
        wl["params"].get("depth", 0),
        wl["num_qubits"],
        wl["shots"],
        str(wl["params"].get("layout")),
    )


def _heat_panel(ax, order, workloads, cols, cells, norm, title, width, show_rows=True) -> None:
    ax.grid(False)
    ax.set_xlim(0, width)
    ax.set_ylim(len(order), 0)
    for i, wid in enumerate(order):
        base = cells.get((wid, "prismq"))
        for j, name in enumerate(cols):
            r = cells.get((wid, name))
            text, face, hatch = "", SURFACE, None
            if r is None:
                text, hatch = "not run", "////"
            elif base is None or _status(base) != "ok":
                text, hatch = "no base", "////"
            elif _status(r) != "ok":
                text, hatch = _status(r), "////"
            else:
                ratio = stats.ratio(r["samples_ms"], base["samples_ms"])["ratio"]
                face = DIVERGING(norm(math.log10(ratio)))
                text = f"{ratio:.2f}x" if ratio < 100 else f"{ratio:.0f}x"
            ax.add_patch(
                plt.Rectangle(
                    (j + 0.03, i + 0.05),
                    0.94,
                    0.9,
                    facecolor=face,
                    edgecolor=GRID if hatch else "none",
                    hatch=hatch,
                    linewidth=0.6,
                )
            )
            lum = 1.0 if hatch else sum(face[:3]) / 3
            ax.text(
                j + 0.5,
                i + 0.5,
                text,
                ha="center",
                va="center",
                fontsize=7.5,
                color=MUTED if hatch else ("#ffffff" if lum < 0.45 else INK),
            )
    ax.set_xticks([j + 0.5 for j in range(len(cols))])
    ax.set_xticklabels([_label(c) for c in cols], rotation=30, ha="right")
    ax.set_yticks([i + 0.5 for i in range(len(order))])
    ax.set_yticklabels([_row_label(workloads[w]) for w in order] if show_rows else [])
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, loc="left")


def heatmap(records: list[dict], out_dir: str, written: list[str]) -> None:
    rows = [r for r in records if r["group"] == "headline"]
    workloads: dict[str, dict] = {}
    cells: dict[tuple[str, str], dict] = {}
    top = defaultdict(int)
    for r in rows:
        top[r["workload"]["workload_id"]] = max(top[r["workload"]["workload_id"]], r["threads"])
    for r in rows:
        if r["threads"] == top[r["workload"]["workload_id"]]:
            workloads[r["workload"]["workload_id"]] = r["workload"]
            cells[(r["workload"]["workload_id"], r["simulator"])] = r
    present = {name for (_, name) in cells}

    def block(mode: str, cols: list[str]) -> tuple[list[str], list[str]]:
        cols = [c for c in cols if c in present]
        order = [w for w in workloads if workloads[w]["mode"] == mode and any((w, c) in cells for c in cols)]
        return sorted(order, key=lambda k: _family_order(workloads[k])), cols

    dense64, cols64 = block("probs", ["aer-statevector", "quest", "spinoza", "qip"])
    dense32, cols32 = block("probs", ["aer-statevector-f32", "qsim"])
    struct, cols_s = block("counts", ["stim", "aer-mps", "aer-automatic"])
    dense = sorted(set(dense64) | set(dense32), key=lambda k: _family_order(workloads[k]))
    if not dense and not struct:
        return
    width = max(len(cols64), len(cols_s), 1)
    fig = plt.figure(figsize=(1.05 * (width + len(cols32)) + 3.6, 0.3 * (len(dense) + len(struct)) + 3.2))
    gs = fig.add_gridspec(
        2,
        3,
        height_ratios=[max(1, len(dense)), max(1, len(struct))],
        width_ratios=[width, max(1, len(cols32)), 0.25],
        hspace=0.45,
        wspace=0.08,
    )
    norm = TwoSlopeNorm(vmin=-2.0, vcenter=0.0, vmax=2.0)
    if dense:
        _heat_panel(
            fig.add_subplot(gs[0, 0]),
            dense,
            workloads,
            cols64,
            cells,
            norm,
            "Dense probability output: fp64 comparators",
            width,
        )
        if cols32:
            _heat_panel(
                fig.add_subplot(gs[0, 1]),
                dense,
                workloads,
                cols32,
                cells,
                norm,
                "fp32 comparators",
                len(cols32),
                show_rows=False,
            )
    if struct:
        _heat_panel(
            fig.add_subplot(gs[1, 0]),
            struct,
            workloads,
            cols_s,
            cells,
            norm,
            "Structured circuits, sampled output",
            width,
        )
    cax = fig.add_subplot(gs[:, 2])
    sm = plt.cm.ScalarMappable(norm=norm, cmap=DIVERGING)
    bar = fig.colorbar(sm, cax=cax)
    bar.set_ticks([-2, -1, 0, 1, 2])
    bar.set_ticklabels(["0.01x", "0.1x", "1x", "10x", "100x"])
    bar.outline.set_visible(False)
    bar.set_label("comparator median / PRISM-Q median", color=INK_2)
    fig.suptitle(
        "Time ratio against PRISM-Q, per workload", x=0.01, ha="left", fontsize=11, fontweight="semibold", color=INK
    )
    _save(fig, out_dir, "fig1-ratio-heatmap", written)


def _by_family(records: list[dict], group: str) -> dict[str, list[dict]]:
    fam: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        if r["group"] != group:
            continue
        wl = r["workload"]
        key = wl["family"]
        if wl["family"] in ("rcs", "lowent"):
            key += f" d{wl['params'].get('depth')}"
        if wl["family"] == "disjoint" and wl["params"].get("layout") == "strided":
            continue
        if wl["family"] == "midcircuit":
            key += f" {_shots(wl['shots'])}"
        fam[key].append(r)
    return fam


def size_scaling(
    records: list[dict], out_dir: str, written: list[str], metric: str, stem: str, title: str, ylabel: str
) -> None:
    fam = _by_family(records, "headline")
    fam = {k: v for k, v in fam.items() if len({r["workload"]["num_qubits"] for r in v}) >= 2}
    if not fam:
        return
    keys = sorted(fam, key=lambda k: _family_order(fam[k][0]["workload"]))
    cols = 4
    rows_n = math.ceil(len(keys) / cols)
    fig, axes = plt.subplots(rows_n, cols, figsize=(3.2 * cols, 2.7 * rows_n), squeeze=False)
    handles: dict[str, Any] = {}
    for ax, key in zip(axes.flat, keys):
        series: dict[str, list[tuple]] = defaultdict(list)
        missing: list[tuple] = []
        for r in fam[key]:
            n = r["workload"]["num_qubits"]
            st = _status(r)
            if st == "ok":
                if metric == "time":
                    s = r["stats"]
                    series[r["simulator"]].append((n, s["median_ms"], s["p25_ms"], s["p75_ms"]))
                elif r.get("peak_rss_mb"):
                    series[r["simulator"]].append((n, r["peak_rss_mb"], r["peak_rss_mb"], r["peak_rss_mb"]))
            elif st != "n/a":
                missing.append((n, r["simulator"], st))
        for name, pts in sorted(series.items(), key=lambda kv: kv[0] != "prismq"):
            pts.sort()
            xs, ys, lo, hi = zip(*pts)
            h = ax.errorbar(
                xs,
                ys,
                yerr=[np.subtract(ys, lo), np.subtract(hi, ys)],
                color=_color(name),
                marker="o",
                markersize=4.5,
                linewidth=1.6 if name != "prismq" else 2.2,
                capsize=0,
                elinewidth=1.0,
                label=_label(name),
                zorder=3 if name == "prismq" else 2,
            )
            handles.setdefault(_label(name), h)
        ax.set_yscale("log")
        ax.set_title(key, loc="left")
        ax.set_xlabel("qubits")
        ax.set_ylabel(ylabel)
        qubits = sorted({r["workload"]["num_qubits"] for r in fam[key]})
        ax.set_xticks(qubits)
        if max(qubits) / max(1, min(qubits)) > 8:
            ax.set_xscale("log")
            ax.set_xticks(qubits)
            ax.set_xticklabels([str(q) for q in qubits])
            ax.minorticks_off()
        if missing:
            ymax = ax.get_ylim()[1]
            for n, name, st in missing:
                ax.scatter([n], [ymax], marker="x", color=_color(name), s=30, zorder=4, clip_on=False)
            ax.text(
                0.99,
                0.02,
                "x at top: " + ", ".join(sorted({f"{_label(m)} {s}" for _, m, s in missing})),
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=6.5,
                color=INK_2,
                wrap=True,
            )
    for ax in axes.flat[len(keys) :]:
        ax.set_visible(False)
    fig.legend(
        handles.values(), handles.keys(), loc="lower center", ncol=min(6, len(handles)), bbox_to_anchor=(0.5, -0.02)
    )
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, fontweight="semibold", color=INK)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    _save(fig, out_dir, stem, written)


def absolute(records: list[dict], out_dir: str, written: list[str]) -> None:
    """Dot plot of every headline workload at the headline thread count, IQR as a bar.

    Dense-output rows and sampled rows sit in separate panels: they run on
    different simulator sets, and each panel's colours are unique within it.
    """
    rows = [r for r in records if r["group"] == "headline" and r["status"] == "ok" and r.get("samples_ms")]
    if not rows:
        return
    panels = [
        ("Dense probability output", [r for r in rows if r["workload"]["mode"] == "probs"]),
        ("Sampled output", [r for r in rows if r["workload"]["mode"] == "counts"]),
    ]
    panels = [(t, p) for t, p in panels if p]
    heights = [len({r["workload"]["workload_id"] for r in p}) for _, p in panels]
    fig, axes = plt.subplots(
        len(panels),
        1,
        figsize=(8.0, 0.3 * sum(heights) + 2.2 * len(panels)),
        gridspec_kw={"height_ratios": heights},
        squeeze=False,
    )
    for ax, (title, prow) in zip(axes[:, 0], panels):
        workloads = {r["workload"]["workload_id"]: r["workload"] for r in prow}
        order = sorted(workloads, key=lambda k: _family_order(workloads[k]))
        sims = sorted({r["simulator"] for r in prow}, key=lambda s: (s != "prismq", s))
        offsets = {s: (i - (len(sims) - 1) / 2) * (0.7 / max(1, len(sims))) for i, s in enumerate(sims)}
        for r in prow:
            y = order.index(r["workload"]["workload_id"]) + offsets[r["simulator"]]
            st = r["stats"]
            invalid = _status(r) == "invalid"
            ax.plot([st["p25_ms"], st["p75_ms"]], [y, y], color=_color(r["simulator"]), linewidth=1.2)
            ax.scatter(
                [st["median_ms"]],
                [y],
                s=22,
                color=SURFACE if invalid else _color(r["simulator"]),
                edgecolor=_color(r["simulator"]),
                linewidth=1.2,
                zorder=3,
            )
        ax.set_xscale("log")
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([_row_label(workloads[w]) for w in order])
        ax.set_ylim(len(order) - 0.5, -0.5)
        ax.set_xlabel("median wall time per call, ms (bar: p25 to p75; open marker: output failed validation)")
        for name in sims:
            ax.scatter([], [], color=_color(name), label=_label(name), s=22)
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
        ax.set_title(title, loc="left")
    fig.suptitle(
        "Absolute execution time, headline thread count",
        x=0.01,
        ha="left",
        fontsize=11,
        fontweight="semibold",
        color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    _save(fig, out_dir, "fig2-absolute-times", written)


def strong_scaling(records: list[dict], out_dir: str, written: list[str]) -> None:
    rows = aggregate.scaling(records)
    if not rows:
        return
    by_wl: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row["speedup_vs_1"] is not None:
            by_wl[row["workload_id"]][row["simulator"]].append((row["threads"], row["speedup_vs_1"]))
    if not by_wl:
        return
    fig, axes = plt.subplots(1, len(by_wl), figsize=(3.3 * len(by_wl), 3.0), squeeze=False)
    handles = {}
    for ax, (wid, sims) in zip(axes[0], sorted(by_wl.items())):
        top = max(t for pts in sims.values() for t, _ in pts)
        ax.plot([1, top], [1, top], color=AXIS, linewidth=1.0, label="linear")
        for name, pts in sorted(sims.items(), key=lambda kv: kv[0] != "prismq"):
            pts.sort()
            xs, ys = zip(*pts)
            (h,) = ax.plot(xs, ys, marker="o", color=_color(name), label=_label(name))
            handles.setdefault(_label(name), h)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=2)
        ticks = sorted({t for pts in sims.values() for t, _ in pts})
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks])
        ax.set_yticks([1, 2, 4, 8])
        ax.set_yticklabels(["1x", "2x", "4x", "8x"])
        ax.minorticks_off()
        ax.set_xlabel("threads")
        ax.set_ylabel("speedup over 1 thread")
        ax.set_title(wid.split("/seed")[0].replace("/probs", ""), loc="left", fontsize=9)
    fig.legend(handles.values(), handles.keys(), loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, -0.05))
    fig.suptitle(
        "Strong scaling (4 physical cores, 8 logical)", x=0.01, ha="left", fontsize=11, fontweight="semibold", color=INK
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    _save(fig, out_dir, "fig4-strong-scaling", written)


def dispatch_figure(records: list[dict], out_dir: str, written: list[str]) -> None:
    rows = [r for r in records if r["group"] == "dispatch"]
    if not rows:
        return
    workloads = {r["workload"]["workload_id"]: r["workload"] for r in rows}
    order = sorted(workloads, key=lambda k: _family_order(workloads[k]))
    fig, ax = plt.subplots(figsize=(8.0, 0.75 * len(order) + 1.4))
    finite = [r["stats"]["median_ms"] for r in rows if _status(r) == "ok"]
    xmax = max(finite) * 3 if finite else 10
    for i, wid in enumerate(order):
        recs = [r for r in rows if r["workload"]["workload_id"] == wid and r["simulator"].startswith("prismq")]
        failed = []
        labelled = 0
        for r in sorted(recs, key=lambda r: (r.get("stats") or {}).get("median_ms") or 0):
            st = _status(r)
            auto = r["simulator"] == "prismq"
            if st == "ok":
                x = r["stats"]["median_ms"]
                ax.scatter(
                    [x],
                    [i],
                    s=60 if auto else 26,
                    color=_color("prismq") if auto else DE_EMPHASIS,
                    zorder=3 if auto else 2,
                    edgecolor=SURFACE,
                    linewidth=1.0,
                )
                if not auto:
                    rise = 6 + 8 * (labelled % 2)
                    labelled += 1
                    ax.annotate(
                        r["simulator"].split(":")[1],
                        (x, i),
                        xytext=(0, rise),
                        textcoords="offset points",
                        ha="center",
                        fontsize=6.5,
                        color=INK_2,
                        arrowprops={"arrowstyle": "-", "color": AXIS, "linewidth": 0.5} if rise > 6 else None,
                    )
            else:
                failed.append(r["simulator"].split(":")[-1] if not auto else "auto")
        auto = next((r for r in recs if r["simulator"] == "prismq"), None)
        if auto is not None and auto.get("resolved_backend"):
            ax.annotate(
                f"auto -> {auto['resolved_backend']}",
                (auto["stats"].get("median_ms") or xmax, i),
                xytext=(0, -11),
                textcoords="offset points",
                ha="center",
                fontsize=7,
                color=_color("prismq"),
            )
        if failed:
            ax.text(
                1.01,
                i,
                "declined or timed out: " + ", ".join(sorted(failed)),
                transform=ax.get_yaxis_transform(),
                va="center",
                fontsize=6.5,
                color=MUTED,
            )
    ax.set_xscale("log")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([_row_label(workloads[w]) for w in order])
    ax.set_ylim(len(order) - 0.5, -0.7)
    ax.set_xlabel("median wall time per call, ms")
    ax.set_title("Automatic dispatch (blue) against every backend forced explicitly (gray)", loc="left")
    _save(fig, out_dir, "fig6-dispatch", written)


def accuracy(records: list[dict], out_dir: str, written: list[str]) -> None:
    rows = [r for r in records if r["group"] == "accuracy" and _status(r) in ("ok", "invalid")]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    series: dict[str, list] = defaultdict(list)
    floor = 1e-16
    for r in rows:
        v = r.get("validation") or {}
        if v.get("tvd") is None or r["simulator"] == v.get("reference"):
            continue
        base = r["simulator"].partition(":")[0]
        option = r["simulator"].partition(":")[2]
        bond = option.split("-")[-1] if option.startswith("mps") or base == "aer-mps" else "exact"
        series[base].append((r["stats"]["median_ms"], max(v["tvd"], floor), bond))
    for name, pts in sorted(series.items(), key=lambda kv: kv[0] != "prismq"):
        mps_pts = sorted([p for p in pts if p[2] != "exact"], key=lambda p: int(p[2]))
        if mps_pts:
            xs, ys, bonds = zip(*mps_pts)
            ax.plot(
                xs, ys, marker="o", color=_color(name), label=f"{SHORT['prismq'] if name == 'prismq' else 'Aer'} MPS"
            )
            for x, y, b in mps_pts:
                ax.annotate(f"chi {b}", (x, y), xytext=(4, 3), textcoords="offset points", fontsize=6.5, color=INK_2)
        for x, y, b in pts:
            if b == "exact":
                ax.scatter([x], [y], marker="s", color=_color(name), s=30, label=f"{_label(name)} statevector")
    ax.axhline(floor, color=AXIS, linewidth=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("median wall time per call, ms")
    ax.set_ylabel("TVD against the exact distribution")
    ax.set_title("Accuracy against runtime under a bond-dimension cap", loc="left")
    ax.legend(loc="upper right")
    _save(fig, out_dir, "fig7-accuracy-runtime", written)


def breakdown(records: list[dict], out_dir: str, written: list[str]) -> None:
    rows = [
        r
        for r in records
        if r["group"] == "headline" and _status(r) == "ok" and (r.get("phases") or {}).get("first_ms")
    ]
    if not rows:
        return
    by_wl: dict[str, list] = defaultdict(list)
    for r in rows:
        by_wl[r["workload"]["workload_id"]].append(r)
    picks = sorted(by_wl, key=lambda k: (-len(by_wl[k]), -by_wl[k][0]["workload"]["num_qubits"]))[:4]
    picks = sorted(picks, key=lambda k: _family_order(by_wl[k][0]["workload"]))
    parts = [
        ("Python start and imports", "#c3c2b7"),
        ("circuit build", "#86b6ef"),
        ("cold-call excess", "#eb6834"),
        ("warm call (median)", "#2a78d6"),
    ]
    fig, axes = plt.subplots(
        len(picks), 1, figsize=(7.5, 1.1 * sum(len(by_wl[k]) for k in picks) * 0.42 + 1.4), squeeze=False
    )
    for ax, wid in zip(axes[:, 0], picks):
        recs = sorted(by_wl[wid], key=lambda r: (r["simulator"] != "prismq", r["simulator"]))
        for i, r in enumerate(recs):
            ph = r["phases"]
            warm = r["stats"]["median_ms"]
            # Only measured phases: the Rust runner does not time its own start,
            # and its process wall time also holds output hashing and file writes.
            startup = (ph.get("startup_ms") or 0) + (ph.get("import_ms") or 0)
            values = [startup, ph.get("build_ms") or 0, max(0.0, ph["first_ms"] - warm), warm]
            left = 0.0
            for (name, color), value in zip(parts, values):
                if value > 0:
                    ax.barh(i, value, left=left, color=color, height=0.62, edgecolor=SURFACE, linewidth=1.0)
                left += value
            ax.text(left, i, f"  {left:.0f} ms", va="center", fontsize=7, color=INK_2)
        ax.set_yticks(range(len(recs)))
        ax.set_yticklabels([_label(r["simulator"]) for r in recs])
        ax.invert_yaxis()
        ax.set_title(_row_label(recs[0]["workload"]), loc="left", fontsize=9)
        ax.set_xlabel("ms: Python start (Python-driven simulators only), build, first call, one warm call")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c in parts]
    fig.legend(handles, [n for n, _ in parts], loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Where the time goes in one process", x=0.01, ha="left", fontsize=11, fontweight="semibold", color=INK)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    _save(fig, out_dir, "fig8-phase-breakdown", written)


def lowest_ratios(records: list[dict], out_dir: str, written: list[str]) -> None:
    """The validated comparisons with the smallest ratios: every one at or below 1.25x, at least twelve."""
    rows = [x for x in aggregate.ratios(records) if x.get("ratio") and x["validated"]]
    if not rows:
        return
    rows.sort(key=lambda x: x["ratio"])
    keep = [x for x in rows if x["ratio"] <= 1.25]
    if len(keep) < 12:
        keep = rows[:12]
    norm = TwoSlopeNorm(vmin=-2.0, vcenter=0.0, vmax=2.0)
    fig, ax = plt.subplots(figsize=(7.5, 0.3 * len(keep) + 1.3))
    for i, x in enumerate(keep):
        value = math.log10(x["ratio"])
        ax.barh(i, value, color=DIVERGING(norm(value)), height=0.6)
        lo, hi = x.get("ci95_low"), x.get("ci95_high")
        if lo and hi:
            ax.plot([math.log10(lo), math.log10(hi)], [i, i], color=INK_2, linewidth=0.9)
        ax.text(max(value, 0) + 0.03, i, f"{x['ratio']:.2f}x", va="center", fontsize=7, color=INK_2)
    ax.axvline(0, color=INK_2, linewidth=0.8)
    ax.set_yticks(range(len(keep)))
    ax.set_yticklabels(
        [f"{x['workload_id'].split('/seed')[0]} vs {_label(x['comparator'])} @{x['threads']}t" for x in keep],
        fontsize=7,
    )
    ax.invert_yaxis()
    ticks = [-1, -0.5, 0, 0.5, 1]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{10**t:.2g}x" for t in ticks])
    ax.set_xlabel("comparator median / PRISM-Q median (line: bootstrap 95% interval)")
    ax.set_title("Smallest time ratios among validated comparisons", loc="left")
    _save(fig, out_dir, "fig9-lowest-ratios", written)


def write_all(results_dir: str, out_dir: str | None = None) -> list[str]:
    _style()
    _, records = aggregate.load(results_dir)
    out_dir = out_dir or os.path.join(results_dir, "figures")
    os.makedirs(out_dir, exist_ok=True)
    written: list[str] = []
    heatmap(records, out_dir, written)
    absolute(records, out_dir, written)
    size_scaling(
        records,
        out_dir,
        written,
        "memory",
        "fig3-peak-memory",
        "Peak resident memory by circuit size",
        "peak working set, MB",
    )
    strong_scaling(records, out_dir, written)
    size_scaling(
        records,
        out_dir,
        written,
        "time",
        "fig5-size-scaling",
        "Execution time by circuit size (bar: p25 to p75)",
        "median ms",
    )
    dispatch_figure(records, out_dir, written)
    accuracy(records, out_dir, written)
    breakdown(records, out_dir, written)
    lowest_ratios(records, out_dir, written)
    print(f"wrote {len(written)} figure files to {out_dir}")
    return written
