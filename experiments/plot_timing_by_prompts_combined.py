#!/usr/bin/env python3
"""1x4 timing figure: LoVR / VideoChapters / MSR-VTT bars + cross-dataset scatter.

Panel size matches the original 1x3 figure by scaling width:
  original figsize=(12.5, 4.2) for 3 panels
  -> figsize=(12.5 * 4/3, 4.2) for 4 panels

Reads experiments/data/<dataset>.log and writes:
  experiments/figures/timing_by_prompts_combined.pdf
"""

from __future__ import annotations

import os
import re
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
OUT_DIR = HERE / "figures"

MODULES = ["Parser", "Candidate Scenes Generator", "Time Boundary Detection"]
MODULE_COLORS = {
    "Parser": "#9e9e9e",
    "Candidate Scenes Generator": "#2a9d8f",
    "Time Boundary Detection": "#e9c46a",
}
HATCHES = {
    "Parser": "\\\\",
    "Candidate Scenes Generator": "xx",
    "Time Boundary Detection": "//",
}
DISPLAY = {
    "lovr": "LoVR",
    "videochapter": "VideoChapters",
    "msrvtt": "MSR-VTT",
}
# Bar-panel order (left three).
BAR_ORDER = ["videochapter", "msrvtt", "lovr"]
# Scatter point order / legend order.
SCATTER_ORDER = ["msrvtt", "lovr", "videochapter"]
SCATTER_MARKERS = {"lovr": "o", "msrvtt": "s", "videochapter": "D"}
SCATTER_COLORS = {
    "lovr": "#7EB8DA",
    "msrvtt": "#2a9d8f",
    "videochapter": "#e76f51",
}

# Original 1x3 used figsize=(12.5, 4.2); keep the same per-panel canvas.
N_PANELS_ORIG = 3
FIGSIZE_ORIG = (12.5, 4.2)
N_PANELS = 4
FIGSIZE = (FIGSIZE_ORIG[0] * N_PANELS / N_PANELS_ORIG, FIGSIZE_ORIG[1])

EDGE_LW = 1.5
FONTSIZE = 12

QUERY_RE = re.compile(r"^Query\s+(\d+):\s+'(.*)'$")
STEP_RE = re.compile(r"^\s*Step\s+\d+\s+\[Conf:")
TIME_LINE_RE = re.compile(r"^\s*(.+?):\s+([-+]?\d*\.?\d+)s\s*$")
NUM_PREFIX_RE = re.compile(r"^\d+(?:\.\d+)*\.?\s+")
PROMPTS_RE = re.compile(r"Total prompts to score \(All clusters\):\s+(\d+)")

ALIASES = {
    "Query Decomposition (LLM)": "query_decomp",
    "Complex Query Embedding": "complex_emb",
    "Complex Query Search": "complex_search",
    "Sub-queries Embedding": "sub_emb",
    "Sub-queries Search": "sub_search",
    "Sub-queries Processing": "sub_proc",
    "Hard time-order filter": "hard_filter",
    "Build bridge inputs": "prompt_build",
    "Write bridge input": "bridge_write",
    "Bridge inference": "bridge_infer",
    "Read bridge output": "bridge_read",
    "Build candidates": "matching",
    "Total Query Processing Time": "total",
}
SKIP = {
    "Intersection & Ranking",
    "Retrieval Processing Time",
    "Cluster query",
    "Frame query",
    "Image decode/save",
}


def _empty_timing() -> dict:
    return {k: 0.0 for k in set(ALIASES.values())}


def module_times(t: dict) -> dict:
    return {
        "Parser": t["query_decomp"],
        "Candidate Scenes Generator": (
            t["complex_emb"]
            + t["complex_search"]
            + t["sub_emb"]
            + t["sub_search"]
            + t["sub_proc"]
            + t["hard_filter"]
        ),
        "Time Boundary Detection": (
            t["prompt_build"]
            + t["bridge_write"]
            + t["bridge_infer"]
            + t["bridge_read"]
            + t["matching"]
        ),
        "vlm_scoring": (
            t["prompt_build"]
            + t["bridge_write"]
            + t["bridge_infer"]
            + t["bridge_read"]
        ),
    }


def parse_dataset_log(log_path: Path) -> pd.DataFrame:
    rows = []
    cur = None
    in_breakdown = False

    def flush():
        nonlocal cur
        if cur is None:
            return
        mods = module_times(cur["t"])
        rows.append(
            {
                "query_number": cur["n"],
                "n_subqueries": cur["l"],
                "n_prompts": cur.get("n_prompts", 0),
                **mods,
            }
        )
        cur = None

    for line in log_path.read_text(encoding="utf-8").splitlines():
        q_match = QUERY_RE.match(line)
        if q_match:
            flush()
            cur = {
                "n": int(q_match.group(1)),
                "l": 0,
                "t": _empty_timing(),
                "n_prompts": 0,
            }
            in_breakdown = False
            continue
        if cur is None:
            continue
        if STEP_RE.match(line) and not in_breakdown:
            cur["l"] += 1
            continue
        p_match = PROMPTS_RE.search(line)
        if p_match:
            cur["n_prompts"] = int(p_match.group(1))
            continue
        if line.strip() == "--- Execution Time Breakdown ---":
            in_breakdown = True
            continue
        if not in_breakdown:
            continue
        t_match = TIME_LINE_RE.match(line)
        if not t_match:
            continue
        label = NUM_PREFIX_RE.sub("", t_match.group(1).strip()).strip()
        value = float(t_match.group(2))
        if label in SKIP:
            continue
        key = ALIASES.get(label)
        if key is not None:
            cur["t"][key] = value

    flush()
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("query_number").reset_index(drop=True)
    return df


def _style_axes(ax: plt.Axes) -> None:
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(EDGE_LW)
    ax.tick_params(axis="x", length=0, which="both", labelsize=FONTSIZE)
    ax.tick_params(
        axis="y", direction="in", length=4, which="both", labelsize=FONTSIZE
    )
    ax.yaxis.grid(True, linestyle="--", linewidth=0.8, color="#b0b0b0", alpha=0.85)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)


def _draw_bar_panel(ax: plt.Axes, df: pd.DataFrame, ds: str, ylabel: bool):
    d = df.sort_values("n_prompts").reset_index(drop=True)
    x = np.arange(len(d))
    bottom = np.zeros(len(d))
    bars_for_legend = []

    for mod in MODULES:
        vals = d[mod].to_numpy()
        bars = ax.bar(
            x,
            vals,
            bottom=bottom,
            color=MODULE_COLORS[mod],
            edgecolor="#333333",
            linewidth=EDGE_LW,
            hatch=HATCHES[mod],
            label=mod,
            width=0.72,
        )
        bars_for_legend.append(bars)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(
        [str(int(p)) for p in d["n_prompts"]],
        rotation=45,
        ha="right",
        fontsize=FONTSIZE,
    )
    ax.set_xlabel("#VLM prompts", fontsize=FONTSIZE)
    ax.set_title(DISPLAY[ds], fontsize=FONTSIZE, pad=8)
    if ylabel:
        ax.set_ylabel("Time (s)", fontsize=FONTSIZE)

    _style_axes(ax)

    ymax = float(bottom.max()) if len(bottom) else 1.0
    if ds == "lovr":
        ylim_top = 49
        ax.set_ylim(0, ylim_top)
        ax.set_yticks([10, 20, 30, 40])
    else:
        ylim_top = ymax * 1.28
        ax.set_ylim(0, ylim_top)
        if ds == "videochapter":
            yticks = list(range(30, int(np.floor(ylim_top)) + 1, 30))
            ax.set_yticks(yticks)

    panel_idx = BAR_ORDER.index(ds) + 1
    for xi, yi, qn in zip(x, bottom, d["query_number"]):
        ax.text(
            xi,
            yi + ymax * 0.02,
            f"q{panel_idx}.{int(qn)}",
            ha="center",
            va="bottom",
            rotation=30,
            fontsize=FONTSIZE,
            color="#444",
        )
    return [b[0] for b in bars_for_legend]


def _add_break_marks(fig: plt.Figure, ax_left: plt.Axes, ax_right: plt.Axes) -> None:
    """Draw // break marks with fixed physical size in the gap (figure coords).

    Using axes-fraction deltas makes marks look different on wide vs narrow
    segments; figure-inch sizing keeps both breaks identical (like the narrow
    right-hand break the user prefers).
    """
    pos_l = ax_left.get_position()
    pos_r = ax_right.get_position()
    xmid = 0.5 * (pos_l.x1 + pos_r.x0)
    fw, fh = fig.get_size_inches()
    # ~0.07 inch slash length; ~0.035 inch separation between the two strokes.
    dx = 0.07 / fw
    dy = 0.07 / fh
    ox = 0.035 / fw
    kwargs = dict(
        color="#333333",
        lw=EDGE_LW,
        clip_on=False,
        transform=fig.transFigure,
        solid_capstyle="butt",
        zorder=6,
    )
    for y in (pos_l.y0, pos_l.y1):
        for o in (-ox / 2, ox / 2):
            fig.add_artist(
                plt.Line2D(
                    [xmid + o - dx, xmid + o + dx],
                    [y - dy, y + dy],
                    **kwargs,
                )
            )


def _draw_scatter_broken(
    fig: plt.Figure,
    panel_ax: plt.Axes,
    dfs: dict[str, pd.DataFrame],
):
    """Replace panel_ax with a 3-segment broken-x scatter.

    Compresses empty mid-ranges so the dense low-prompt cluster gets more width:
      [0, 850] | [1000, 1350] | [2350, xmax]
    """
    pos = panel_ax.get_position()
    panel_ax.remove()

    # Width fractions favoring the dense left cluster.
    fracs = (0.58, 0.22, 0.20)
    gap = 0.014 * pos.width
    usable = pos.width - 2 * gap
    widths = [usable * f for f in fracs]
    x0 = pos.x0
    axes_seg = []
    for i, w in enumerate(widths):
        axes_seg.append(fig.add_axes([x0, pos.y0, w, pos.height]))
        x0 += w + (gap if i < len(widths) - 1 else 0)
    ax_l, ax_m, ax_r = axes_seg

    handles = []
    xs_all = []
    for ds in SCATTER_ORDER:
        g = dfs[ds]
        for ax in axes_seg:
            sc = ax.scatter(
                g["n_prompts"],
                g["vlm_scoring"],
                s=100,
                marker=SCATTER_MARKERS[ds],
                color=SCATTER_COLORS[ds],
                label=DISPLAY[ds],
                alpha=0.9,
                zorder=3,
                edgecolors="#333333",
                linewidths=EDGE_LW * 0.8,
            )
        handles.append(sc)
        xs_all.append(g["n_prompts"].to_numpy(dtype=float))

    x = np.concatenate(xs_all)
    x_max = float(x.max())
    xlims = [(0.0, 850.0), (1000.0, 1350.0), (2350.0, x_max * 1.03)]
    y_min = 0.0
    y_max = 90.0
    y_ticks = [20, 40, 60, 80]

    for ax, (x0l, x1l) in zip(axes_seg, xlims):
        ax.set_xlim(x0l, x1l)
        ax.set_ylim(y_min, y_max)
        ax.set_yticks(y_ticks)

    # Spine / tick styling for a broken axis chain.
    for ax in axes_seg:
        for side in ("top", "bottom", "left", "right"):
            ax.spines[side].set_visible(True)
            ax.spines[side].set_linewidth(EDGE_LW)
        ax.tick_params(axis="x", length=0, which="both", labelsize=FONTSIZE)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        ax.yaxis.grid(
            True, linestyle="--", linewidth=0.8, color="#b0b0b0", alpha=0.85
        )
        ax.xaxis.grid(False)
        ax.set_axisbelow(True)

    ax_l.spines["right"].set_visible(False)
    ax_m.spines["left"].set_visible(False)
    ax_m.spines["right"].set_visible(False)
    ax_r.spines["left"].set_visible(False)

    ax_l.tick_params(
        axis="y", direction="in", length=4, which="both", labelsize=FONTSIZE
    )
    for ax in (ax_m, ax_r):
        ax.tick_params(
            axis="y", direction="in", length=0, which="both", labelleft=False
        )

    ax_l.set_ylabel("Fine-grained scoring time (s)", fontsize=FONTSIZE)
    # Center x-label under the full broken-axis panel (all three segments).
    fig.text(
        pos.x0 + 0.5 * pos.width,
        pos.y0 - 0.12,
        "#VLM prompts",
        ha="center",
        va="top",
        fontsize=FONTSIZE,
        transform=fig.transFigure,
    )

    _add_break_marks(fig, ax_l, ax_m)
    _add_break_marks(fig, ax_m, ax_r)
    return handles, axes_seg


def plot_combined(dfs: dict[str, pd.DataFrame], out_path: Path) -> None:
    fig, axes = plt.subplots(1, N_PANELS, figsize=FIGSIZE, sharey=False)
    print(f"figsize={FIGSIZE}  (orig 1x3 was {FIGSIZE_ORIG})")

    module_handles = None
    for ax, ds in zip(axes[:3], BAR_ORDER):
        handles = _draw_bar_panel(ax, dfs[ds], ds, ylabel=(ax is axes[0]))
        if module_handles is None:
            module_handles = handles

    # Placeholder panel; replaced by broken-x axes after layout alignment.
    axes[3].set_xlabel("#VLM prompts", fontsize=FONTSIZE)
    axes[3].set_ylabel("Fine-grained scoring time (s)", fontsize=FONTSIZE)
    _style_axes(axes[3])

    # Leave headroom for legends above subplot titles; bottom room for (a)/(b).
    fig.tight_layout(rect=[0, 0.06, 1, 0.84])
    fig.canvas.draw()

    # Align x-axis baselines: same y0/height for all four panels.
    ref = axes[0].get_position()
    y0 = min(ax.get_position().y0 for ax in axes[:3])
    height = ref.height
    for ax in axes:
        pos = ax.get_position()
        ax.set_position([pos.x0, y0, pos.width, height])

    scatter_handles, scatter_axes = _draw_scatter_broken(fig, axes[3], dfs)

    bar_left = axes[0].get_position()
    bar_right = axes[2].get_position()
    scatter_pos = scatter_axes[0].get_position()
    scatter_right = scatter_axes[-1].get_position()
    y_legend = bar_left.y1 + 0.055

    module_x = 0.5 * (bar_left.x0 + bar_right.x1)
    scatter_x = 0.5 * (scatter_pos.x0 + scatter_right.x1)

    fig.legend(
        module_handles,
        MODULES,
        loc="lower center",
        bbox_to_anchor=(module_x, y_legend),
        bbox_transform=fig.transFigure,
        ncol=3,
        frameon=False,
        fontsize=FONTSIZE,
        handlelength=2.2,
        columnspacing=1.8,
    )
    fig.legend(
        scatter_handles,
        [DISPLAY[ds] for ds in SCATTER_ORDER],
        loc="lower center",
        bbox_to_anchor=(scatter_x, y_legend),
        bbox_transform=fig.transFigure,
        ncol=3,
        frameon=False,
        fontsize=FONTSIZE,
        handlelength=1.2,
        columnspacing=0.7,
        handletextpad=0.35,
        borderpad=0.15,
        labelspacing=0.2,
    )

    # Group captions under the bar trio (a) and the scatter panel (b).
    y_caption = bar_left.y0 - 0.175
    fig.text(
        module_x,
        y_caption,
        "(a) End-to-end time breakdown of SOLVED.",
        ha="center",
        va="top",
        fontsize=FONTSIZE,
        transform=fig.transFigure,
    )
    fig.text(
        scatter_x,
        y_caption,
        "(b) Fine-grained scoring time of SOLVED.",
        ha="center",
        va="top",
        fontsize=FONTSIZE,
        transform=fig.transFigure,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def main():
    dfs = {}
    for ds in BAR_ORDER:
        path = DATA_DIR / f"{ds}.log"
        if not path.exists():
            raise FileNotFoundError(f"Missing log: {path}")
        dfs[ds] = parse_dataset_log(path)
        print(
            f"{ds}: {len(dfs[ds])} queries | "
            f"prompts={dfs[ds]['n_prompts'].tolist()}"
        )

    out = OUT_DIR / "timing_by_prompts_combined.pdf"
    plot_combined(dfs, out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
