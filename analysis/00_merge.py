#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
00_merge.py
Robust merger for Tecplot ASCII (FENSAP-style):
- Merges all wall zones (titles containing "WALL") from base file into one closed 2D polyline (z <= threshold).
- Augments with variables from auxiliary files (same wall zone titles, identical node indexing).
- For zones missing in an auxiliary file, appended columns are NaN.
- Coordinates with z > threshold + tol are excluded entirely (never written).

Usage:
  python 00_merge.py soln.fensap.000001.dat \
    --augment droplet.drop.000001.dat --augment-prefix droplet \
    --augment swimsol.ice.000001.dat --augment-prefix swim \
    --out merged.dat --z-threshold 0.0 --tolerance 0.0

Author: ChatGPT (2025-08-16)
"""
from __future__ import annotations
import argparse, re, sys, tempfile, zipfile
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
from types import SimpleNamespace

SURFACE_ZONETYPES = {"FEQUADRILATERAL", "FETRIANGLE"}

def _normalize(name: str) -> str:
    name = name.strip()
    name = re.split(r"[\s(;]", name, 1)[0]
    name = re.sub(r"[^A-Za-z0-9]", "", name)
    return name.lower()

def _get_var_index(var_map: Dict[str,int], candidates: List[str]) -> int:
    for cand in candidates:
        idx = var_map.get(_normalize(cand))
        if idx is not None: return idx
    raise KeyError(f"Variable not found among candidates: {candidates}")

def _load_text(path: Path) -> List[str]:
    return Path(path).read_text(encoding="utf-8", errors="replace").splitlines()

def _parse_variables(lines: List[str]) -> Tuple[List[str], Dict[str,int]]:
    var_line = next((ln for ln in lines if ln.lstrip().upper().startswith("VARIABLES")), "")
    if not var_line:
        raise ValueError("VARIABLES line not found")
    var_names = re.findall(r'"([^"]+)"', var_line)
    var_map = {_normalize(v): i for i, v in enumerate(var_names)}
    return var_names, var_map

def _zone_headers(lines: List[str]) -> List[int]:
    return [i for i, ln in enumerate(lines) if ln.lstrip().startswith("ZONE")]

def _read_zone_payload(lines: List[str], start: int, end: int, n_vars: int) -> Tuple[np.ndarray, Optional[np.ndarray], dict]:
    """Return node_vals, conn (or None), and header info dict with keys: title, ztype, N, E"""
    header = lines[start]
    # Join a few continuation header lines (rarely used, but safe)
    header_ext = header
    for look_ahead in range(start + 1, min(end, start + 8)):
        nxt = lines[look_ahead].strip()
        # stop if we hit numeric data or quoted strings
        if nxt.startswith('"') or re.match(r'^[\s\+\-]?\d', nxt):
            break
        header_ext += " " + nxt

    title = (re.search(r'T="([^"]+)', header_ext).group(1) if re.search(r'T="([^"]+)', header_ext) else "")
    ztype = (re.search(r"ZONETYPE=([^,\s]+)", header_ext).group(1).upper() if re.search(r"ZONETYPE=([^,\s]+)", header_ext) else "")
    mN = re.search(r"N=\s*(\d+)", header_ext)
    mE = re.search(r"E=\s*(\d+)", header_ext)
    N = int(mN.group(1)) if mN else None
    E = int(mE.group(1)) if mE else 0

    text = " ".join(line.strip() for line in lines[start+1:end])
    text = re.sub(r"(?<=\d)([+-]\d{2,})", r"e\1", text)  # fix 1.23+05
    values = np.fromstring(text, sep=" ")

    if N is None:
        if n_vars == 0 or values.size == 0 or (values.size % n_vars) != 0:
            raise ValueError("Cannot infer N for zone")
        N = int(values.size // n_vars)
        E = 0

    node_vals = values[: N * n_vars].reshape(N, n_vars)
    conn = None
    if E and ztype in SURFACE_ZONETYPES:
        nnpe = 4 if ztype == "FEQUADRILATERAL" else 3
        if values.size >= N * n_vars + E * nnpe:
            raw = values[N * n_vars : N * n_vars + E * nnpe].reshape(E, nnpe).astype(int) - 1
            # reduce faces to boundary edges (each unique undirected edge with count==1)
            edge_counts: Dict[Tuple[int,int], int] = {}
            for elem in raw:
                verts = [int(n) for n in elem]
                verts = verts + [verts[0]]
                for a, b in zip(verts, verts[1:]):
                    if a == b: continue
                    e = (a,b) if a < b else (b,a)
                    edge_counts[e] = edge_counts.get(e,0) + 1
            edges = [e for e,c in edge_counts.items() if c == 1]
            conn = np.array(edges, dtype=int) if edges else None

    info = {"title": title, "ztype": ztype, "N": N, "E": E}
    return node_vals, conn, info

def read_solution_simple(path: Path, z_thr: float, tol: float):
    lines = _load_text(path)
    var_names, var_map = _parse_variables(lines)
    z_idx = _get_var_index(var_map, ["z"])
    starts = _zone_headers(lines) + [len(lines)]
    wall_zones: List[SimpleNamespace] = []
    for idx,(start,end) in enumerate(zip(starts, starts[1:]), start=1):
        if start >= len(lines): break
        node_vals, conn, info = _read_zone_payload(lines, start, end, len(var_names))
        title_lower = (info["title"] or "").lower()
        header = lines[start].lower()
        is_wall = ("wall" in title_lower) or (" wall" in header) or ("surface" in header) or ("solid" in header)
        if not is_wall: continue
        # filter z
        mask = node_vals[:, z_idx] <= (z_thr + tol)
        nodes = node_vals[mask]
        # remap connectivity if present
        elem = None
        if conn is not None and mask.any():
            idx_map = {old: new for new, old in enumerate(np.where(mask)[0])}
            new_edges = []
            for a,b in conn:
                if mask[a] and mask[b]:
                    new_edges.append([idx_map[a], idx_map[b]])
            elem = np.array(new_edges, dtype=int) if new_edges else None
        wall_zones.append(SimpleNamespace(title=info["title"], nodes=nodes, elem=elem))
    return wall_zones, var_names, var_map

def walk_order(z: SimpleNamespace, x_idx: int, y_idx: int) -> np.ndarray:
    # Prefer connectivity; if absent, assume file order
    if z.elem is None or z.elem.size == 0:
        return np.arange(len(z.nodes), dtype=int)
    # Build adjacency & walk deterministically
    from collections import defaultdict
    adj: Dict[int, List[int]] = defaultdict(list)
    for a,b in z.elem:
        a=int(a); b=int(b)
        adj[a].append(b); adj[b].append(a)
    # find endpoints
    endpoints = [n for n,nb in adj.items() if len(nb)==1]
    start = min(endpoints) if endpoints else min(adj.keys())
    order = [start]; used = set()
    cur = start
    while True:
        nbs = adj[cur]
        nxt = None
        for n in nbs:
            e = tuple(sorted((cur,n)))
            if e not in used:
                used.add(e)
                nxt = n; break
        if nxt is None or nxt==start: break
        order.append(nxt); cur = nxt
    return np.array(order, dtype=int)

def merge_with_map(walls: List[SimpleNamespace], var_map: Dict[str,int]) -> Tuple[np.ndarray, np.ndarray, List[Tuple[str,int]]]:
    x_idx = _get_var_index(var_map, ["x"])
    y_idx = _get_var_index(var_map, ["y"])
    nodes_list = []
    orders = []
    titles = []
    offsets = []
    edges = []
    off = 0
    prev_last_xy = None
    for z in walls:
        ord_local = walk_order(z, x_idx, y_idx)
        nodes_ord = z.nodes[ord_local]
        # orient to connect to previous
        if prev_last_xy is not None and len(nodes_ord)>0:
            d_start = np.linalg.norm(nodes_ord[0,[x_idx,y_idx]] - prev_last_xy)
            d_end   = np.linalg.norm(nodes_ord[-1,[x_idx,y_idx]] - prev_last_xy)
            if d_end < d_start:
                nodes_ord = nodes_ord[::-1]
                ord_local = ord_local[::-1]
        n = len(nodes_ord)
        nodes_list.append(nodes_ord)
        orders.append(ord_local)
        titles.append(getattr(z,'title',''))
        if n>1:
            edges.append(np.column_stack([np.arange(off, off+n-1), np.arange(off+1, off+n)]))
        if nodes_ord.size:
            prev_last_xy = nodes_ord[-1,[x_idx,y_idx]]
        offsets.append(off); off += n
    if nodes_list:
        all_nodes = np.concatenate(nodes_list, axis=0)
    else:
        all_nodes = np.empty((0, len(var_map)), dtype=float)
    if edges:
        conn = np.concatenate(edges, axis=0)
        # close the loop
        if all_nodes.shape[0] >= 2:
            conn = np.vstack([conn, [all_nodes.shape[0]-1, 0]])
    else:
        conn = np.empty((0,2), dtype=int)

    # mapping global idx -> (title, local original index)
    map_list = []
    # reconstruct per-zone linear index to original index
    # per zone, ord_local maps local-linear -> original-index
    g = 0
    for title, ord_local in zip(titles, orders):
        for li, orig in enumerate(ord_local):
            map_list.append((title, int(orig)))
            g += 1
    return all_nodes, conn, map_list

def write_tecplot(path: Path, nodes: np.ndarray, conn: np.ndarray, var_names: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    var_line = " ".join(f'"{v}"' for v in var_names)
    with open(path, "w", encoding="utf-8") as f:
        f.write('TITLE = "Merged Wall Data"\n')
        f.write(f"VARIABLES = {var_line}\n")
        n_nodes = nodes.shape[0]; n_elem = conn.shape[0]
        f.write(f'ZONE T="MergedWall", N={n_nodes}, E={n_elem}, DATAPACKING=POINT, ZONETYPE=FELINESEG\n')
        for row in nodes:
            f.write(" ".join(str(v) for v in row) + "\n")
        for a,b in conn:
            f.write(f"{int(a)+1} {int(b)+1}\n")

def read_dat_or_zip(p: Path):
    if p.suffix.lower() != ".zip":
        return p, None
    tmp = tempfile.TemporaryDirectory()
    with zipfile.ZipFile(p, "r") as zf:
        zf.extractall(tmp.name)
    # pick the first .dat
    dats = sorted(Path(tmp.name).rglob("*.dat"))
    if not dats: raise FileNotFoundError(f"No .dat in {p}")
    return dats[0], tmp

def main():
    ap = argparse.ArgumentParser(description="Merge wall zones and augment variables across files (z<=threshold).")
    ap.add_argument("solution", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--z-threshold", type=float, default=0.0)
    ap.add_argument("--tolerance", type=float, default=0.0)
    ap.add_argument("--augment", action="append", metavar="FILE")
    ap.add_argument("--augment-prefix", action="append", metavar="NAME")
    ap.add_argument("--merge-only", action="store_true")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    base_path, tmp_base = read_dat_or_zip(args.solution)
    try:
        walls, base_var_names, base_var_map = read_solution_simple(base_path, args.z_threshold, args.tolerance)
        if not walls:
            raise SystemExit("No wall zones detected in base solution. Try adjusting --z-threshold/--tolerance.")
        nodes, conn, merge_map = merge_with_map(walls, base_var_map)
        base_nodes = nodes.copy()
        out_var_names = list(base_var_names)

        if args.augment:
            prefixes = args.augment_prefix if args.augment_prefix else [Path(a).stem for a in args.augment]
            if len(prefixes) != len(args.augment):
                raise ValueError("--augment-prefix must have same length as --augment")
            for aug, pref in zip(args.augment, prefixes):
                aug_path, tmp_aug = read_dat_or_zip(Path(aug))
                try:
                    awalls, avars, amap = read_solution_simple(aug_path, args.z_threshold, args.tolerance)
                    # map title -> nodes
                    a_by_title = {(getattr(z,'title','') or '').strip().lower(): z.nodes for z in awalls}
                    # choose columns to add: those not already in base (skip x,y,z)
                    base_keys = {_normalize(v) for v in out_var_names}
                    a_keys_norm = [_normalize(v) for v in avars]
                    skip = {"x","y","z"}
                    new_keys = [k for k in a_keys_norm if (k not in base_keys) and (k not in skip)]
                    if not new_keys: continue
                    new_cols_idx = [ {_normalize(v):i for i,v in enumerate(avars) }[k] for k in new_keys ]
                    add = np.full((base_nodes.shape[0], len(new_cols_idx)), np.nan, dtype=float)
                    for i,(ztitle, lidx) in enumerate(merge_map):
                        zkey = (ztitle or "").strip().lower()
                        zmat = a_by_title.get(zkey)
                        if zmat is None: continue
                        if 0 <= lidx < zmat.shape[0]:
                            add[i,:] = zmat[lidx, new_cols_idx]
                    add_names = [f"{pref}:{avars[ {_normalize(v):i for i,v in enumerate(avars)}[k] ]}" for k in new_keys]
                    base_nodes = np.column_stack([base_nodes, add])
                    out_var_names.extend(add_names)
                finally:
                    if tmp_aug is not None: tmp_aug.cleanup()

        write_tecplot(args.out, base_nodes, conn, out_var_names)
    finally:
        if tmp_base is not None: tmp_base.cleanup()

if __name__ == "__main__":
    main()
