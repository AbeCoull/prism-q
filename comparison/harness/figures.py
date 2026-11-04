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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
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


def _save(fig, out_dir: str, stem: str, written: list[str]) -> None:
    path = os.path.join(out_dir, f"{stem}.svg")
    fig.savefig(path, bbox_inches="tight", dpi=200, metadata={"Date": None})
    written.append(path)
    plt.close(fig)


def _status(record: dict) -> str:
    if record["status"] != "ok":
        return {"not-applicable": "n/a"}.get(record["status"], record["status"])
    v = record.get("validation") or {}
    if v.get("pass") is False:
        return "invalid"
    return "ok"


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
    ax.set_xticklabels([aggregate.label(c) for c in cols], rotation=30, ha="right")
    ax.set_yticks([i + 0.5 for i in range(len(order))])
    ax.set_yticklabels([aggregate.row_label(workloads[w]) for w in order] if show_rows else [])
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
        return sorted(order, key=lambda k: aggregate.family_order(workloads[k])), cols

    dense64, cols64 = block("probs", ["aer-statevector", "quest", "spinoza", "qip"])
    dense32, cols32 = block("probs", ["aer-statevector-f32", "qsim"])
    struct, cols_s = block("counts", ["stim", "aer-mps", "aer-automatic"])
    dense = sorted(set(dense64) | set(dense32), key=lambda k: aggregate.family_order(workloads[k]))
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
    _save(fig, out_dir, "ratio-heatmap", written)


def strong_scaling(records: list[dict], out_dir: str, written: list[str]) -> None:
    rows = aggregate.scaling(records)
    if not rows:
        return
    workloads = {r["workload"]["workload_id"]: r["workload"] for r in records if r["group"] == "threads"}
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
            (h,) = ax.plot(xs, ys, marker="o", color=_color(name), label=aggregate.label(name))
            handles.setdefault(aggregate.label(name), h)
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
        ax.set_title(aggregate.row_label(workloads[wid]), loc="left", fontsize=9)
    fig.legend(handles.values(), handles.keys(), loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, -0.05))
    fig.suptitle(
        "Strong scaling (4 physical cores, 8 logical)", x=0.01, ha="left", fontsize=11, fontweight="semibold", color=INK
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    _save(fig, out_dir, "thread-scaling", written)


def dispatch_figure(records: list[dict], out_dir: str, written: list[str]) -> None:
    rows = [r for r in records if r["group"] == "dispatch"]
    if not rows:
        return
    workloads = {r["workload"]["workload_id"]: r["workload"] for r in rows}
    order = sorted(workloads, key=lambda k: aggregate.family_order(workloads[k]))
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
    ax.set_yticklabels([aggregate.row_label(workloads[w]) for w in order])
    ax.set_ylim(len(order) - 0.5, -0.7)
    ax.set_xlabel("median wall time per call, ms")
    ax.set_title("Automatic dispatch (blue) against every backend forced explicitly (gray)", loc="left")
    _save(fig, out_dir, "dispatch", written)


def write_all(results_dir: str, out_dir: str | None = None) -> list[str]:
    _style()
    _, records = aggregate.load(results_dir)
    out_dir = out_dir or os.path.join(results_dir, "figures")
    os.makedirs(out_dir, exist_ok=True)
    written: list[str] = []
    heatmap(records, out_dir, written)
    strong_scaling(records, out_dir, written)
    dispatch_figure(records, out_dir, written)
    print(f"wrote {len(written)} figure files to {out_dir}")
    return written
