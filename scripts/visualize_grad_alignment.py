# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Visualize step_{t}_grad_alignment.csv files produced by the online probe,
in a classic ML-paper (seaborn) style.

For every step_{t}_grad_alignment.csv found in --csv_dir, writes:
  step_{t}_heatmap.png   - layer (rows, 0..L-1) x {attn, ffn} heatmap of cos(g_RL, g_WM)

Across all steps found, writes three combined line plots (one curve per step):
  layer_dot.png          - cos(g_RL, g_WM) vs layer
  layer_norm_rl.png      - ||g_RL|| vs layer
  layer_norm_wm.png      - ||g_WM|| vs layer

Usage:
    python scripts/visualize_grad_alignment.py \
        --csv_dir outputs/results/grad_alignment/<project>/<exp>
"""

import argparse
import glob
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

DPI = 400
LAYER_RE = re.compile(r"^layer_(\d+)$")

DOT_LABEL = r"$\cos(g_{\text{RL}}, g_{\text{WM}})$"
NORM_RL_LABEL = r"$\|g_{\text{RL}}\|_2$"
NORM_WM_LABEL = r"$\|g_{\text{WM}}\|_2$"
LAYER_LABEL = r"$L$"
STEP_LABEL = r"$t$"

sns.set_theme(style="ticks", context="paper")
plt.rcParams.update({
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "font.serif": ["cmr10", "Computer Modern Roman", "DejaVu Serif"],
    "axes.unicode_minus": False,
    "font.size": 15,
    "axes.labelsize": 18,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 15,
    "legend.title_fontsize": 16,
    "axes.formatter.use_mathtext": True,
})


def parse_args():
    p = argparse.ArgumentParser(description="Visualize grad-alignment CSVs (seaborn, paper style)")
    p.add_argument("--csv_dir", type=str, required=True, help="Dir containing step_{t}_grad_alignment.csv files")
    p.add_argument("--output_dir", type=str, default=None, help="Defaults to --csv_dir")
    return p.parse_args()


def find_step_csvs(csv_dir):
    paths = glob.glob(os.path.join(csv_dir, "step_*_grad_alignment.csv"))
    return sorted(paths, key=lambda p: int(re.search(r"step_(\d+)", os.path.basename(p)).group(1)))


def layer_index(layer_str):
    m = LAYER_RE.match(layer_str)
    return int(m.group(1)) if m else None


def plot_heatmap(df, step, out_path, cosine_limit):
    sub = df[df["param_type"].isin(["attn", "ffn"]) & df["layer"].str.match(r"^layer_\d+$")].copy()
    sub["layer_idx"] = sub["layer"].apply(layer_index)
    pivot = sub.pivot(index="layer_idx", columns="param_type", values="cosine")
    pivot = pivot.reindex(index=sorted(pivot.index), columns=["attn", "ffn"])

    fig, ax = plt.subplots(figsize=(4.2, 6.4))
    sns.heatmap(
        pivot,
        cmap=sns.diverging_palette(240, 10, as_cmap=True),
        center=0, vmin=-cosine_limit, vmax=cosine_limit,
        linewidths=0,
        square=False,
        xticklabels=[r"$\mathrm{attn}$", r"$\mathrm{ffn}$"],
        yticklabels=[f"${i}$" for i in pivot.index],
        cbar_kws={"label": DOT_LABEL, "shrink": 0.55, "aspect": 22, "pad": 0.05},
        ax=ax,
    )
    ax.set_ylabel(LAYER_LABEL, fontsize=18)
    ax.set_xlabel("")
    ax.tick_params(axis="x", labelsize=15, bottom=False)
    ax.tick_params(axis="y", labelsize=13, rotation=0, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = ax.collections[0].colorbar
    cbar.outline.set_linewidth(0.5)
    cbar.ax.tick_params(labelsize=13)
    cbar.set_label(DOT_LABEL, fontsize=17)

    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


def build_combined(dfs_with_steps):
    rows = []
    for df, step in dfs_with_steps:
        sub = df[(df["param_type"] == "ALL") & df["layer"].str.match(r"^layer_\d+$")].copy()
        sub["layer_idx"] = sub["layer"].apply(layer_index)
        sub["step"] = step
        rows.append(sub[["step", "layer_idx", "cosine", "norm_rl", "norm_wm"]])
    return pd.concat(rows, ignore_index=True).sort_values(["step", "layer_idx"])


def plot_metric_lines(combined, value_col, ylabel, out_path):
    steps = sorted(combined["step"].unique())
    cmap = matplotlib.colormaps["viridis"]
    norm = matplotlib.colors.Normalize(vmin=steps[0], vmax=steps[-1])
    palette = {step: cmap(norm(step)) for step in steps}

    fig, ax = plt.subplots(figsize=(6.8, 3.8), layout="constrained")
    sns.lineplot(
        data=combined,
        x="layer_idx",
        y=value_col,
        hue="step",
        palette=palette,
        hue_order=steps,
        legend=False,
        linewidth=2,
        marker="o",
        markersize=6,
        markeredgewidth=0,
        ax=ax,
    )
    ax.set_xlabel(LAYER_LABEL, fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)

    if value_col == "cosine":
        ax.axhline(0, color="0.5", linewidth=0.8, linestyle="--")

    if len(steps) > 1:
        # Use the same numeric step-to-color mapping for lines and colorbar.
        tick_count = min(6, len(steps))
        ticks = [steps[round(i * (len(steps) - 1) / (tick_count - 1))] for i in range(tick_count)]
        cbar = fig.colorbar(
            matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
            ax=ax, ticks=ticks, fraction=0.045, pad=0.04, aspect=24,
        )
        cbar.ax.set_title(STEP_LABEL, fontsize=16, pad=8)
        cbar.ax.tick_params(labelsize=12)
        cbar.outline.set_linewidth(0.5)
    else:
        ax.text(0.98, 0.98, rf"$t={steps[0]}$", transform=ax.transAxes, ha="right", va="top")
    if not combined[value_col].notna().any():
        ax.text(0.5, 0.5, "No finite measurements", transform=ax.transAxes, ha="center")

    sns.despine(ax=ax)
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


def plot_layer_steps(combined, dfs_with_steps, output_dir):
    """Plot pooled layer cosine over training steps on the fixed [-1, 1] scale."""
    values = combined.pivot(index="layer_idx", columns="step", values="cosine").sort_index()
    steps = list(values.columns)
    global_cosines = []
    for df, _ in dfs_with_steps:
        pooled = df[(df["layer"] == "ALL") & (df["param_type"] == "ALL")]
        global_cosines.append(float(pooled["cosine"].iloc[0]) if len(pooled) else np.nan)
    fig, axes = plt.subplots(
        2, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [4, 1]}, layout="constrained"
    )
    im = axes[0].imshow(
        values.to_numpy(), origin="lower", aspect="auto", interpolation="nearest",
        cmap="RdBu_r", vmin=-1.0, vmax=1.0,
    )
    axes[0].set_xticks(range(len(steps)), steps)
    indices = sorted(set(list(range(0, len(values), 5)) + [len(values) - 1]))
    axes[0].set_yticks(indices, [values.index[i] for i in indices])
    axes[0].set_xlabel(STEP_LABEL)
    axes[0].set_ylabel(LAYER_LABEL)
    fig.colorbar(im, ax=axes[0], label=DOT_LABEL)
    axes[1].plot(steps, global_cosines, "-o", color="#31688e", markersize=4)
    axes[1].axhline(0, color="gray", linewidth=0.8)
    axes[1].set_xlabel(STEP_LABEL)
    axes[1].set_ylabel("Global cosine")
    axes[1].grid(alpha=0.2)
    for extension in ("png", "pdf"):
        path = os.path.join(output_dir, f"layer_step_cosine.{extension}")
        fig.savefig(path, dpi=DPI)
        print(f"Wrote {path}")
    plt.close(fig)


def main():
    args = parse_args()
    csv_paths = find_step_csvs(args.csv_dir)
    if not csv_paths:
        raise FileNotFoundError(f"No step_*_grad_alignment.csv files found in {args.csv_dir}")
    output_dir = args.output_dir or args.csv_dir
    os.makedirs(output_dir, exist_ok=True)

    dfs_with_steps = []
    for path in csv_paths:
        df = pd.read_csv(path)
        step = int(re.search(r"step_(\d+)", os.path.basename(path)).group(1))
        if df.empty:
            print(f"Skipped step {step}: no observation targets")
            continue
        dfs_with_steps.append((df, step))

    if not dfs_with_steps:
        print("No measured steps to plot")
        return
    # One zero-centered scale makes colors comparable between training steps.
    finite_cosines = pd.concat([df["cosine"] for df, _ in dfs_with_steps]).dropna()
    cosine_limit = max(float(finite_cosines.abs().max()), 1e-6) if len(finite_cosines) else 1.0
    for df, step in dfs_with_steps:
        heatmap_path = os.path.join(output_dir, f"step_{step}_heatmap.png")
        plot_heatmap(df, step, heatmap_path, cosine_limit)
        print(f"Wrote {heatmap_path}")

    combined = build_combined(dfs_with_steps)
    plot_layer_steps(combined, dfs_with_steps, output_dir)

    dot_path = os.path.join(output_dir, "layer_dot.png")
    plot_metric_lines(combined, "cosine", DOT_LABEL, dot_path)
    print(f"Wrote {dot_path}")

    norm_rl_path = os.path.join(output_dir, "layer_norm_rl.png")
    plot_metric_lines(combined, "norm_rl", NORM_RL_LABEL, norm_rl_path)
    print(f"Wrote {norm_rl_path}")

    norm_wm_path = os.path.join(output_dir, "layer_norm_wm.png")
    plot_metric_lines(combined, "norm_wm", NORM_WM_LABEL, norm_wm_path)
    print(f"Wrote {norm_wm_path}")


if __name__ == "__main__":
    main()
