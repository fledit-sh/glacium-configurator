#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_auto_cp_normals.py — Cp aus INLET_1000 (Zeile mit min(x)) bestimmen und Normalenplot auf merged wall erzeugen.

Beispiele:
  # Auto: p_inf/q_inf aus solution, merge ausführen, Plot als PNG
  python 01_auto_cp_normals.py \
      --solution solutions.zip \
      --merged-out merged.dat \
      --png-out merged_normals.png

  # Wenn merged.dat schon existiert:
  python 01_auto_cp_normals.py --solution soln.fensap.000001.dat --merged-in merged.dat --png-out merged_normals.png

Hinweise:
- 'INLET_1000' wird zonenweise fall-insensitiv gesucht. Alternativ kann --inlet-name gesetzt werden.
- p_inf wird aus der min(x)-Linie gemittelt.
- q_inf wird bevorzugt aus 'q'/'q_inf'/'dynamicpressure' Variablen gemittelt;
  fehlt das, dann aus rho und |U| (Velocity Magnitude oder Komponenten u,v,[w]).
- Die Normalen zeigen robust "nach außen" (von der Profilfläche weg).
"""

from __future__ import annotations
import argparse, re, sys, zipfile, tempfile, subprocess, math
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt

# --------------- Generic Tecplot parsing helpers ---------------

def _norm_key(s: str) -> str:
    return re.sub(r'[^A-Za-z0-9]', '', s).lower()

def _find_var_exact(var_names: List[str], *cands: Optional[str]) -> Optional[int]:
    norm = {_norm_key(v): i for i, v in enumerate(var_names)}
    for c in [c for c in cands if c is not None]:
        i = norm.get(_norm_key(c))
        if i is not None:
            return i
    return None

def _find_var_fuzzy(var_names: List[str], *needles: str) -> Optional[int]:
    keys = [_norm_key(v) for v in var_names]
    for n in needles:
        if not n:
            continue
        nn = _norm_key(n)
        for i, k in enumerate(keys):
            if k == nn:
                return i
        for i, k in enumerate(keys):
            if nn in k:
                return i
    return None

def _read_all_text(path: Path) -> str:
    return path.read_text(encoding='utf-8', errors='ignore')

def _read_until_numbers(f, count: int) -> np.ndarray:
    vals = []
    while len(vals) < count:
        line = f.readline()
        if not line:
            break
        for p in line.strip().split():
            try:
                vals.append(float(p))
            except ValueError:
                pass
    return np.asarray(vals, dtype=float)

def parse_first_zone(path: Path) -> Tuple[np.ndarray, List[str], Optional[np.ndarray], str]:
    """Parse the first (merged wall) zone of a Tecplot ASCII file with FELINESEG connectivity possible.
    Returns nodes [N x nvar], var_names, conn [E x 2] or None, and header blob for later info.
    """
    with path.open('r', encoding='utf-8', errors='ignore') as f:
        head_lines = []
        for _ in range(10000):
            pos = f.tell()
            line = f.readline()
            if not line:
                break
            head_lines.append(line)
            if re.match(r'^\s*[-+0-9.]', line):
                f.seek(pos)
                break
        head = ''.join(head_lines)

        m_vars = re.search(r'VARIABLES\s*=\s*(.+)', head, re.I)
        if not m_vars:
            raise RuntimeError("VARIABLES= nicht gefunden (merged).")
        var_blob = m_vars.group(1)
        raw = re.findall(r'"([^"]+)"|\'([^\']+)\'|([A-Za-z0-9_\.]+)', var_blob)
        var_names = [next(filter(None, t)) for t in raw]

        mN = re.search(r'\bN\s*=\s*(\d+)', head, re.I)
        if not mN:
            raise RuntimeError("N= nicht gefunden (merged).")
        N = int(mN.group(1))

        mE = re.search(r'\bE\s*=\s*(\d+)', head, re.I)
        E = int(mE.group(1)) if mE else 0

        mZT = re.search(r'\bZONETYPE\s*=\s*([A-Z0-9_]+)', head, re.I)
        zonetype = mZT.group(1).upper() if mZT else ""

        nvar = len(var_names)
        vals = _read_until_numbers(f, N * nvar)
        if vals.size != N * nvar:
            raise RuntimeError(f"Knotendaten unvollständig ({vals.size} von {N*nvar}).")
        nodes = vals.reshape(N, nvar)

        conn = None
        if E > 0 and 'FELINESEG' in zonetype:
            pairs = []
            while len(pairs) < E:
                line = f.readline()
                if not line:
                    break
                ints = []
                for p in line.strip().split():
                    try:
                        ints.append(int(p))
                    except ValueError:
                        pass
                for i in range(0, len(ints), 2):
                    if i + 1 < len(ints):
                        pairs.append((ints[i]-1, ints[i+1]-1))
            conn = np.array(pairs, dtype=int) if pairs else None
    return nodes, var_names, conn, head

# --------------- Find INLET_1000 zone and extract freestream ---------------

def read_dat_from_zip_or_path(p: Path) -> Path:
    """Return a real .dat path (extract to temp if .zip)."""
    if p.suffix.lower() != '.zip':
        return p
    ztmp = tempfile.TemporaryDirectory()
    with zipfile.ZipFile(p, 'r') as zf:
        zf.extractall(ztmp.name)
    # pick first .dat with INLET in name if exists, else first .dat
    cand = sorted(Path(ztmp.name).rglob("*.dat"))
    if not cand:
        raise FileNotFoundError(f"Keine .dat in {p}")
    # Heuristic: prefer files with 'soln' / 'solution' first
    cand.sort(key=lambda c: (('soln' not in c.name.lower()) and ('solution' not in c.name.lower()), c.name.lower()))
    return cand[0]

def iter_zones(lines: List[str]) -> List[Tuple[int,int,str]]:
    starts = [i for i, ln in enumerate(lines) if ln.strip().upper().startswith("ZONE")]
    starts.append(len(lines))
    zones = []
    for s, e in zip(starts, starts[1:]):
        header = " ".join(lines[s:e][:5])
        zones.append((s, e, header))
    return zones

def parse_variables_from_text(lines: List[str]) -> List[str]:
    var_line = next((ln for ln in lines if ln.lstrip().upper().startswith("VARIABLES")), "")
    if not var_line:
        raise RuntimeError("VARIABLES= nicht gefunden (solution).")
    var_names = re.findall(r'"([^"]+)"|\'([^\']+)\'|([A-Za-z0-9_\.]+)', var_line)
    return [next(filter(None, t)) for t in var_names]

def read_zone_nodes(lines: List[str], start: int, end: int, nvar: int) -> np.ndarray:
    text = " ".join(line.strip() for line in lines[start+1:end])
    text = re.sub(r"(?<=\d)([+-]\d{2,})", r"e\1", text)  # fix 1.23+05
    values = np.fromstring(text, sep=" ")
    if values.size % nvar != 0:
        # try to infer N from header
        mN = re.search(r"\bN\s*=\s*(\d+)", lines[start], re.I)
        if mN:
            N = int(mN.group(1))
            need = N * nvar
            if values.size < need:
                raise RuntimeError("Unvollständige Zonendaten.")
            values = values[:need]
        else:
            raise RuntimeError("Kann Zonendaten nicht parsen.")
    N = values.size // nvar
    return values.reshape(N, nvar)

def extract_freestream_from_inlet(solution: Path, inlet_name: str = "INLET_1000",
                                  x_name: str = "x") -> Tuple[float, float, Dict[str,float]]:
    """Return p_inf, q_inf and a dict of sampled means for reference. Uses min(x)-Row in inlet zone."""
    p = read_dat_from_zip_or_path(solution)
    lines = Path(p).read_text(encoding='utf-8', errors='ignore').splitlines()
    var_names = parse_variables_from_text(lines)
    nvar = len(var_names)
    xi = _find_var_exact(var_names, x_name) or _find_var_fuzzy(var_names, x_name)
    if xi is None:
        raise RuntimeError("X-Variable im solution nicht gefunden.")

    zones = iter_zones(lines)
    target = None
    for s, e, header in zones:
        h = header.lower()
        name = ""
        mT = re.search(r't\s*=\s*"([^"]+)"', header, re.I)
        if mT:
            name = mT.group(1)
        if (inlet_name.lower() in h) or (inlet_name.lower() in name.lower()):
            target = (s, e, header); break
        # Fallback: a zone named "INLET" also acceptable
        if ("inlet" in h) and target is None:
            target = (s, e, header)
    if target is None:
        raise RuntimeError(f"Zone '{inlet_name}' nicht gefunden.")

    s, e, _ = target
    nodes = read_zone_nodes(lines, s, e, nvar)
    x = nodes[:, xi].astype(float)
    xmin = float(np.nanmin(x))
    tol = max(1e-9, 1e-6 * max(1.0, abs(xmin)))
    mask = np.isfinite(x) & (np.abs(x - xmin) <= tol)
    if not np.any(mask):
        # pick absolute argmin
        mask = np.zeros_like(x, dtype=bool)
        mask[int(np.nanargmin(x))] = True

    # ---- determine p_inf ----
    p_candidates = ['p', 'pressure', 'staticpressure', 'pressure(n/m^2)', 'pstat', 'ps']
    pi = _find_var_exact(var_names, *p_candidates) or _find_var_fuzzy(var_names, *p_candidates)
    if pi is None:
        raise RuntimeError("Druckvariable im INLET nicht gefunden.")
    p_inf = float(np.nanmean(nodes[mask, pi]))

    # ---- determine q_inf ----
    # prefer present dynamic pressure variable
    q_candidates = ['q', 'q_inf', 'qinf', 'dynamicpressure', 'dynamic pressure']
    qi = _find_var_exact(var_names, *q_candidates) or _find_var_fuzzy(var_names, *q_candidates)
    if qi is not None:
        q_inf = float(np.nanmean(nodes[mask, qi]))
    else:
        # compute from rho and |U|
        rho_candidates = ['rho', 'density']
        ri = _find_var_exact(var_names, *rho_candidates) or _find_var_fuzzy(var_names, *rho_candidates)
        # velocity magnitude or components
        umag_candidates = ['velocity magnitude', 'umag', 'velmag', 'speed', '|u|']
        ui_mag = _find_var_exact(var_names, *umag_candidates) or _find_var_fuzzy(var_names, *umag_candidates)

        ui = _find_var_exact(var_names, 'u') or _find_var_fuzzy(var_names, 'u')
        vi = _find_var_exact(var_names, 'v') or _find_var_fuzzy(var_names, 'v')
        wi = _find_var_exact(var_names, 'w') or _find_var_fuzzy(var_names, 'w')

        if ri is None or (ui_mag is None and ui is None and vi is None and wi is None):
            raise RuntimeError("q_inf kann nicht bestimmt werden (keine q- oder rho/U-Variablen).")

        rho_inf = float(np.nanmean(nodes[mask, ri]))
        if ui_mag is not None:
            V = float(np.nanmean(nodes[mask, ui_mag]))
        else:
            ux = nodes[mask, ui] if ui is not None else 0.0
            vy = nodes[mask, vi] if vi is not None else 0.0
            wz = nodes[mask, wi] if wi is not None else 0.0
            V = float(np.nanmean(np.sqrt(ux**2 + vy**2 + wz**2)))
        q_inf = 0.5 * rho_inf * V * V

    return p_inf, q_inf, {"xmin": xmin, "p_inf": p_inf, "q_inf": q_inf}

# --------------- Build order and outward normals for merged wall ---------------

def build_order_from_conn(conn: Optional[np.ndarray], N: int) -> np.ndarray:
    if conn is None or conn.size == 0:
        return np.arange(N, dtype=int)
    from collections import defaultdict
    nbr = defaultdict(list); deg = defaultdict(int)
    for a, b in conn:
        a = int(a); b = int(b)
        nbr[a].append(b); nbr[b].append(a)
        deg[a] += 1; deg[b] += 1
    ends = [k for k, v in deg.items() if v == 1]
    start = min(ends) if ends else min(nbr.keys())
    order = [start]; seen = {start}; cur = start; prev = None
    while True:
        nxts = [n for n in nbr[cur] if n != prev and n not in seen]
        if not nxts:
            break
        prev, cur = cur, nxts[0]
        order.append(cur); seen.add(cur)
    return np.asarray(order, dtype=int)

def compute_tangent(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = len(x)
    tx = np.empty(n); ty = np.empty(n)
    for i in range(n):
        if i == 0:
            dx = x[1] - x[0]; dy = y[1] - y[0]
        elif i == n-1:
            dx = x[-1] - x[-2]; dy = y[-1] - y[-2]
        else:
            dx = x[i+1] - x[i-1]; dy = y[i+1] - y[i-1]
        L = np.hypot(dx, dy)
        tx[i], ty[i] = (1.0, 0.0) if L == 0 else (dx / L, dy / L)
    return tx, ty

def _point_in_polygon(xp: float, yp: float, x: np.ndarray, y: np.ndarray) -> bool:
    if x[0] != x[-1] or y[0] != y[-1]:
        x = np.r_[x, x[0]]
        y = np.r_[y, y[0]]
    inside = False
    for i in range(len(x) - 1):
        x1, y1 = x[i], y[i]
        x2, y2 = x[i+1], y[i+1]
        cond = ((y1 > yp) != (y2 > yp))
        if cond:
            xinters = (x2 - x1) * (yp - y1) / (y2 - y1 + 1e-300) + x1
            if xp < xinters:
                inside = not inside
    return inside

def outward_normals(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    tx, ty = compute_tangent(x, y)
    nx = -ty.copy(); ny = tx.copy()
    ds = np.hypot(np.roll(x, -1) - x, np.roll(y, -1) - y)
    if not np.any(ds > 0):
        ds[:] = 1.0
    ds[ds == 0] = np.nanmedian(ds[ds > 0])
    eps = 0.005 * ds
    xc = x if (x[0] == x[-1] and y[0] == y[-1]) else np.r_[x, x[0]]
    yc = y if (x[0] == x[-1] and y[0] == y[-1]) else np.r_[y, y[0]]
    for i in range(len(x)):
        xt = x[i] + eps[i] * nx[i]
        yt = y[i] + eps[i] * ny[i]
        if _point_in_polygon(xt, yt, xc, yc):
            nx[i] = -nx[i]; ny[i] = -ny[i]
    return nx, ny

# --------------- Cp and plotting on merged wall ---------------

def compute_cp_on_wall(nodes: np.ndarray, var_names: List[str], p_inf: float, q_inf: float,
                       p_name: Optional[str]=None, cp_name: Optional[str]=None) -> np.ndarray:
    # Prefer existing Cp column if available
    if cp_name:
        idx = _find_var_exact(var_names, cp_name) or _find_var_fuzzy(var_names, cp_name)
        if idx is not None:
            return nodes[:, idx].astype(float)
    # Else compute from pressure
    p_candidates = [p_name, 'p', 'pressure', 'staticpressure', 'pstat', 'ps', 'pressure(n/m^2)']
    pi = _find_var_exact(var_names, *[c for c in p_candidates if c]) \
         or _find_var_fuzzy(var_names, *[c for c in p_candidates if c])
    if pi is None:
        raise RuntimeError("Druckvariable auf merged wall nicht gefunden.")
    p = nodes[:, pi].astype(float)
    if q_inf == 0.0:
        raise RuntimeError("q_inf ist 0.")
    return (p - float(p_inf)) / float(q_inf)

def plot_cp_normals_png(nodes: np.ndarray, var_names: List[str], conn: Optional[np.ndarray],
                        out_png: Path, cp: np.ndarray,
                        x_name: Optional[str]=None, y_name: Optional[str]=None,
                        base_len: float=0.002, extra_len: float=0.008, dpi: int=300):
    xi = _find_var_exact(var_names, x_name or 'x') or _find_var_fuzzy(var_names, x_name or 'x')
    yi = _find_var_exact(var_names, y_name or 'y') or _find_var_fuzzy(var_names, y_name or 'y')
    if xi is None or yi is None:
        raise RuntimeError("X/Y-Variablen nicht gefunden (merged).")
    x_all = nodes[:, xi].astype(float)
    y_all = nodes[:, yi].astype(float)
    order = build_order_from_conn(conn, len(x_all))
    x = x_all[order]; y = y_all[order]
    cp_ord = cp[order]

    nx, ny = outward_normals(x, y)

    # Länge ∝ |Cp|, normalisiert
    abs_cp = np.abs(cp_ord)
    cpmax = float(np.nanmax(abs_cp)) if np.any(np.isfinite(abs_cp)) else 1.0
    if cpmax <= 0.0: cpmax = 1.0
    scale = abs_cp / cpmax
    mag = base_len + extra_len * scale

    colors = np.empty(len(cp_ord), dtype=object)
    colors[cp_ord < 0.0] = 'tab:blue'
    colors[cp_ord > 0.0] = 'tab:red'
    colors[cp_ord == 0.0] = '0.6'

    plt.figure(figsize=(14, 4.5))
    plt.plot(x, y, 'k-', linewidth=1.0)
    for xi_, yi_, nxi, nyi, mi, ci in zip(x, y, nx, ny, mag, colors):
        plt.plot([xi_, xi_ + nxi * mi], [yi_, yi_ + nyi * mi], '-', linewidth=0.8, color=ci)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title("Cp-Normalen (aus INLET_1000)")
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(out_png, dpi=dpi)
    plt.close()

# --------------- CLI orchestration ---------------

def main():
    ap = argparse.ArgumentParser(description="Cp aus INLET_1000 bestimmen und Normalenplot auf merged wall erzeugen.")
    ap.add_argument("--solution", type=Path, required=True, help=".dat oder .zip mit der Basissolution (enthält INLET_1000)")
    ap.add_argument("--inlet-name", type=str, default="INLET_1000", help="Zonenname für den Inlet (Default: INLET_1000)")
    ap.add_argument("--merged-in", type=Path, default=None, help="Vorhandenes merged.dat (überspringt Merge-Schritt)")
    ap.add_argument("--merged-out", type=Path, default=Path("merged.dat"),
                    help="Ausgabepfad für Merge (wenn --merged-in nicht gesetzt)")
    ap.add_argument("--png-out", type=Path, default=Path("merged_normals.png"), help="Ausgabebild")
    ap.add_argument("--cp-name", type=str, default=None, help="Name der Cp-Variable, falls bereits vorhanden")
    ap.add_argument("--p-name", type=str, default=None, help="Name der Druckvariable (Default auto)")
    ap.add_argument("--x", dest="xname", default=None, help="X-Variablenname (auto)")
    ap.add_argument("--y", dest="yname", default=None, help="Y-Variablenname (auto)")
    ap.add_argument("--base", type=float, default=0.000, help="Basislänge der Normalen")
    ap.add_argument("--extra", type=float, default=0.05, help="Zusatzlänge (Skalierung mit |Cp|)")
    ap.add_argument("--dpi", type=int, default=300, help="DPI")
    ap.add_argument("--z-threshold", type=float, default=0.0, help="z-Filter für WALLs beim Merge")
    ap.add_argument("--tolerance", type=float, default=0.0, help="Toleranz auf z-Filter beim Merge")
    ap.add_argument("--augment", action="append", metavar="FILE", help="Optionale Zusatzdatei(en) zum Mergen (wie 00_merge.py)")
    ap.add_argument("--augment-prefix", action="append", metavar="NAME", help="Präfix(e) für Zusatzvariablen")
    args = ap.parse_args()

    # 1) p_inf, q_inf aus INLET
    p_inf, q_inf, info = extract_freestream_from_inlet(args.solution, args.inlet_name)

    # 2) Merged wall beschaffen (entweder vorhanden oder erzeugen)
    merged_path: Path
    if args.merged_in and args.merged_in.exists():
        merged_path = args.merged_in
    else:
        # Aufruf von 00_merge.py im selben Ordner wie dieses Skript oder in CWD
        merge_py_candidates = [
            Path(__file__).with_name("00_merge.py"),
            Path.cwd() / "00_merge.py"
        ]
        merge_py = next((p for p in merge_py_candidates if p.exists()), None)
        if merge_py is None:
            raise SystemExit("00_merge.py nicht gefunden (neben diesem Skript oder im Arbeitsverzeichnis).")
        cmd = [sys.executable, str(merge_py), str(read_dat_from_zip_or_path(args.solution)),
               "--out", str(args.merged_out),
               "--z-threshold", str(args.z_threshold),
               "--tolerance", str(args.tolerance)]
        if args.augment:
            prefixes = args.augment_prefix if args.augment_prefix else [Path(a).stem for a in args.augment]
            if len(prefixes) != len(args.augment):
                raise SystemExit("--augment-prefix muss gleiche Länge wie --augment haben.")
            for a, pfx in zip(args.augment, prefixes):
                cmd += ["--augment", str(a), "--augment-prefix", str(pfx)]
        print("Merge:", " ".join(cmd))
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print(res.stdout)
            print(res.stderr, file=sys.stderr)
            raise SystemExit("Merge fehlgeschlagen.")
        merged_path = args.merged_out

    # 3) Merged laden
    nodes, var_names, conn, _ = parse_first_zone(merged_path)

    # 4) Cp berechnen (auf merged wall) und plotten
    cp = compute_cp_on_wall(nodes, var_names, p_inf, q_inf, p_name=args.p_name, cp_name=args.cp_name)
    plot_cp_normals_png(nodes, var_names, conn, args.png_out, cp,
                        x_name=args.xname, y_name=args.yname,
                        base_len=args.base, extra_len=args.extra, dpi=args.dpi)

    print(f"p_inf = {p_inf:.6g} Pa, q_inf = {q_inf:.6g} Pa (aus {args.inlet_name}, x_min={info['xmin']:.6g})")
    print(f"Normalen-Plot gespeichert: {args.png_out}")
    print(f"Merged: {merged_path}")

if __name__ == "__main__":
    main()
