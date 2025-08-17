#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_correlations_v4.py
- Reads merged Tecplot ASCII (first ZONE)
- Builds pandas DataFrame with UNIQUE column names (dedup)
- Pearson & Spearman correlations -> CSV + fully labeled heatmaps
- Pairplot on up to top-10 variables (robust against near-constant columns and seaborn bin issues)
"""

import argparse, re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

_num_line_re = re.compile(r'^[\s\+\-]?(?:\d|\.)')

def _parse_variables(lines):
    var_line = next((ln for ln in lines if ln.lstrip().upper().startswith("VARIABLES")), "")
    if not var_line:
        raise ValueError("VARIABLES line not found")
    var_names = re.findall(r'"([^"]+)"', var_line)
    if not var_names:
        var_names = re.findall(r'"([^"]+)"[,\s]*', var_line)
    return var_names

def _is_edge_row(tokens):
    if len(tokens) != 2:
        return False
    return all(re.fullmatch(r"[+\-]?\d+", t) for t in tokens)

def _read_nodes_first_zone(path: Path):
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    var_names = _parse_variables(lines)
    nvars = len(var_names)
    zstarts = [i for i, ln in enumerate(lines) if ln.lstrip().upper().startswith("ZONE")]
    if not zstarts:
        raise ValueError("No ZONE header found")
    start = zstarts[0]
    end = zstarts[1] if len(zstarts) > 1 else len(lines)
    header_ext = lines[start]
    for look_ahead in range(start+1, min(end, start+200)):
        s = lines[look_ahead].strip()
        if _num_line_re.match(s) or s.startswith('"'):
            break
        header_ext += " " + s
    mN = re.search(r"\bN\s*=\s*(\d+)", header_ext, flags=re.I)
    N = int(mN.group(1)) if mN else None
    payload = lines[start+1:end]
    node_rows = []
    for ln in payload:
        s = ln.strip()
        if not s: continue
        toks = s.split()
        if _is_edge_row(toks):
            break
        node_rows.append(s)
    if N is None:
        N = len(node_rows)
    text = " ".join(node_rows)
    text = re.sub(r"(?<=\d)([+-]\d{2,})", r"e\1", text)
    values = np.fromstring(text, sep=" ")
    needed = N * nvars
    if values.size < needed:
        needed = values.size // nvars * nvars
    nodes = values[:needed].reshape(-1, nvars)
    return nodes, var_names

def _dedup(names):
    seen = {}
    out = []
    for n in names:
        if n not in seen:
            seen[n] = 0
            out.append(n)
        else:
            seen[n] += 1
            out.append(f"{n}__{seen[n]}")
    return out

def _select_top_vars_from_corr(corr_df: pd.DataFrame, max_pairs: int = 15, max_vars: int = 10):
    C = corr_df.abs().values
    n = C.shape[0]
    if n < 2:
        return []
    iu = np.triu_indices(n, k=1)
    vals = C[iu]
    order = np.argsort(vals)[::-1]
    chosen_vars = []
    chosen_set = set()
    for idx in order[:max_pairs]:
        i = iu[0][idx]; j = iu[1][idx]
        for k in (i, j):
            if len(chosen_vars) >= max_vars:
                break
            if k not in chosen_set:
                chosen_set.add(k)
                chosen_vars.append(k)
        if len(chosen_vars) >= max_vars:
            break
    cols = [corr_df.columns[i] for i in chosen_vars]
    return cols

def main():
    ap = argparse.ArgumentParser(description="Correlation analysis of merged.dat (robust)")
    ap.add_argument("merged", type=Path)
    ap.add_argument("--outdir", type=Path, default=Path("correlation_analysis"))
    ap.add_argument("--pairplot-max-vars", type=int, default=10)
    ap.add_argument("--pairplot-max-pairs", type=int, default=20)
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    nodes, var_names = _read_nodes_first_zone(args.merged)
    var_names_unique = _dedup(var_names)
    df = pd.DataFrame(nodes, columns=var_names_unique)

    # drop all-NaN and constant columns
    is_all_nan = df.isna().all(axis=0)
    ptp = df.apply(lambda s: (np.nanmax(s.values) - np.nanmin(s.values)) if np.isfinite(s).any() else 0.0)
    is_const = (ptp <= 0.0)
    df = df.loc[:, ~(is_all_nan | is_const)]

    # correlations
    pearson = df.corr(method="pearson", numeric_only=True)
    spearman = df.corr(method="spearman", numeric_only=True)

    pearson.to_csv(args.outdir / "correlation_pearson.csv")
    spearman.to_csv(args.outdir / "correlation_spearman.csv")

    # Heatmaps with all labels (scale fig size with variable count)
    for name, corr in [("pearson", pearson), ("spearman", spearman)]:
        size = max(12, 0.28 * len(corr.columns))
        plt.figure(figsize=(size, size))
        ax = sns.heatmap(corr, cmap="coolwarm", center=0,
                         xticklabels=corr.columns,
                         yticklabels=corr.columns,
                         square=True)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=6)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=6)
        plt.title(f"{name.capitalize()} correlation heatmap", pad=6)
        plt.tight_layout()
        plt.savefig(args.outdir / f"correlation_{name}.png", dpi=300)
        plt.close()

    # Pairplot: pick top variables then filter again for non-degenerate stats
    top_vars = _select_top_vars_from_corr(pearson, max_pairs=args.pairplot_max_pairs, max_vars=args.pairplot_max_vars)
    if len(top_vars) >= 2:
        sub = df[top_vars].copy()
        # Drop columns that became constant within selected subset (after NaN removal)
        sub = sub.dropna()
        nunique = sub.apply(lambda s: np.unique(s.values).size)
        keep = nunique[nunique > 1].index.tolist()
        sub = sub[keep]
        if sub.shape[1] >= 2 and sub.shape[0] >= 10:
            try:
                g = sns.pairplot(sub, corner=True, diag_kind="hist", diag_kws={"bins":"auto"}, plot_kws={"s":10, "linewidth":0})
                g.fig.savefig(args.outdir / "scatter_matrix_top.png", dpi=200)
                plt.close(g.fig)
            except Exception:
                # Fallback to KDE on diagonal if hist binning fails
                g = sns.pairplot(sub, corner=True, diag_kind="kde", plot_kws={"s":10, "linewidth":0})
                g.fig.savefig(args.outdir / "scatter_matrix_top.png", dpi=200)
                plt.close(g.fig)

    print(f"Correlation analysis written to {args.outdir.resolve()}")

if __name__ == "__main__":
    main()
