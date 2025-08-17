#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_x_over_c_v6.py
- Plots ALL variables vs x/c (c=max X) from a merged Tecplot ASCII (first ZONE).
- Friendly with scienceplots. If LaTeX is enabled, labels are made TeX-safe.
- Skips columns that are entirely NaN. Windows-safe filenames.

Usage:
  python plot_x_over_c_v6.py merged.dat --outdir plots
  python plot_x_over_c_v6.py merged.dat --outdir plots --style science
  python plot_x_over_c_v6.py merged.dat --outdir plots --style "science,no-latex"
"""

from __future__ import annotations
import argparse, re
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(["science","ieee"])

_num_line_re = re.compile(r'^[\s\+\-]?(?:\d|\.)')

def _normalize(name: str) -> str:
    name = name.strip()
    name = re.split(r"[\s(;]", name, 1)[0]
    name = re.sub(r"[^A-Za-z0-9]", "", name)
    return name.lower()

def _parse_variables(lines: List[str]) -> Tuple[List[str], Dict[str,int]]:
    var_line = next((ln for ln in lines if ln.lstrip().upper().startswith("VARIABLES")), "")
    if not var_line:
        raise ValueError("VARIABLES line not found")
    var_names = re.findall(r'"([^"]+)"', var_line)
    if not var_names:
        var_names = re.findall(r'"([^"]+)"[,\s]*', var_line)
    var_map = {_normalize(v): i for i, v in enumerate(var_names)}
    return var_names, var_map

def _is_edge_row(tokens: List[str]) -> bool:
    if len(tokens) != 2:
        return False
    return all(re.fullmatch(r"[+\-]?\d+", t) for t in tokens)

def _read_nodes_first_zone(path: Path) -> Tuple[np.ndarray, List[str], Dict[str,int]]:
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    var_names, var_map = _parse_variables(lines)
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
    node_rows: List[str] = []
    for ln in payload:
        s = ln.strip()
        if not s: continue
        toks = s.split()
        if _is_edge_row(toks):
            break
        node_rows.append(s)
    if N is None:
        N = len(node_rows)
        if N == 0:
            raise ValueError("Could not infer N=; no node rows found.")
    text = " ".join(node_rows)
    text = re.sub(r"(?<=\d)([+-]\d{2,})", r"e\1", text)
    values = np.fromstring(text, sep=" ")
    needed = N * nvars
    if values.size < needed:
        raise ValueError(f"Data too small: got {values.size}, need >= {needed}")
    nodes = values[:needed].reshape(N, nvars)
    return nodes, var_names, var_map

def _safe_name(name: str) -> str:
    s = name.strip()
    s = s.replace(":", "__")
    s = re.sub(r'[<>:"/\\|?*]', "_", s)
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "var"

def _latexify_label(name: str) -> str:
    """If LaTeX is on, make common unit patterns safe: m^3 -> m$^3$, underscores escaped, etc."""
    # Escape underscores
    name = name.replace("_", r"\_")
    # Replace carets with superscripts (e.g., m^3, s^-1)
    name = re.sub(r"\^(-?\d+)", r"$^{\1}$", name)
    # Replace per-unit slashes like kg/m^3 -> kg m$^{-3}$ (optional: keep as is)
    return name

def main():
    ap = argparse.ArgumentParser(description="Plot ALL variables vs x/c (c=max X).")
    ap.add_argument("merged", type=Path)
    ap.add_argument("--outdir", type=Path, default=Path("plots_x_over_c"))
    ap.add_argument("--nan-policy", choices=["omit","keep"], default="omit",
                    help="omit: drop NaNs; keep: include NaNs (gaps).")
    ap.add_argument("--style", type=str, default="",
                    help='Matplotlib style(s). Examples: "science", "science,no-latex", "seaborn-v0_8".')
    args = ap.parse_args()

    # Optional style handling (scienceplots friendly)
    if args.style:
        styles = [s.strip() for s in args.style.split(",") if s.strip()]
        try:
            import scienceplots  # noqa: F401
        except Exception:
            pass
        try:
            plt.style.use(styles)
        except Exception as e:
            print(f"[warn] Could not apply style {styles}: {e}")

    nodes, var_names, var_map = _read_nodes_first_zone(args.merged)
    if "x" not in var_map:
        raise SystemExit("X not found in VARIABLES")
    x = nodes[:, var_map["x"]]
    c = float(np.nanmax(x))
    if not np.isfinite(c) or c == 0.0:
        raise SystemExit("max(X) invalid; cannot compute x/c")
    xc = x / c

    use_tex = bool(plt.rcParams.get("text.usetex", False))

    args.outdir.mkdir(parents=True, exist_ok=True)

    for i, name in enumerate(var_names):
        y = nodes[:, i]
        # skip columns that are entirely NaN to avoid empty artifacts
        if np.isnan(y).all():
            continue

        is_y = (_normalize(name) == "y")

        if args.nan_policy == "omit":
            mask = np.isfinite(xc) & np.isfinite(y)
            xs = xc[mask];
            ys = y[mask]
            if xs.size == 0:
                continue
        else:
            xs = xc;
            ys = y

        # scale y as well if this is the Y coordinate
        if is_y:
            ys = ys / c

        # labels/titles
        if use_tex:
            if is_y:
                ylabel = r"y/c"
                title = r"y/c vs x/c"
            else:
                ylabel = _latexify_label(name)
                title = _latexify_label(f"{name} vs x/c")
        else:
            if is_y:
                ylabel = "y/c"
                title = "y/c vs x/c"
            else:
                ylabel = name
                title = f"{name} vs x/c"

        fig, ax = plt.subplots()
        ax.plot(xs, ys)

        # enforce equal aspect ratio only for x/c vs y/c
        if is_y:
            ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x/c")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        try:
            fig.tight_layout()
            fname = f"{_safe_name('y_over_c' if is_y else name)}_vs_x_over_c.png"
            fig.savefig(args.outdir / fname, dpi=150)
        except Exception:
            fname = f"{_safe_name('y_over_c' if is_y else name)}_vs_x_over_c.png"
            fig.savefig(args.outdir / fname, dpi=150, bbox_inches="tight")
        finally:
            plt.close(fig)

    print(f"Done. Plots written to: {args.outdir.resolve()}")

if __name__ == "__main__":
    main()
