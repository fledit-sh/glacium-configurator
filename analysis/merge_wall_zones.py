import re
from pathlib import Path
import argparse
from types import SimpleNamespace

import numpy as np
import matplotlib.pyplot as plt


# Keywords used to automatically identify wall zones.  The search is case
# insensitive and runs against the zone title as well as the raw zone header.
# Additional markers can be added to this list as needed.
WALL_KEYWORDS = ("wall", "surface", "solid")


def _normalize(name: str) -> str:
    """Normalize a Tecplot variable name for lookup."""
    return re.sub(r"\s+", "", name).lower()


def _get_var_index(var_map: dict[str, int], candidates: list[str]) -> int:
    """Return the index of the first matching variable name."""
    for cand in candidates:
        idx = var_map.get(_normalize(cand))
        if idx is not None:
            return idx
    raise KeyError(f"Variable not found among candidates: {candidates}")


def expected_nodes_per_element(zonetype: str) -> int:
    return {
        "FELINESEG": 2,
        "FETRIANGLE": 3,
        "FEQUADRILATERAL": 4,
        "FEPOLYGON": -1,
        "FETETRAHEDRON": 4,
        "FEBRICK": 8,
    }.get((zonetype or "").upper(), -1)


def order_points_from_lineseg(npts: int, elements: np.ndarray) -> list[int]:
    adj = {i: [] for i in range(npts)}
    for a, b in elements:
        a = int(a)
        b = int(b)
        adj[a].append(b)
        adj[b].append(a)
    endpoints = [i for i, n in adj.items() if len(n) == 1]
    start = endpoints[0] if endpoints else 0
    order = [start]
    vis = {start}
    cur = start
    while True:
        nxt = [n for n in adj[cur] if n not in vis]
        if not nxt:
            break
        cur = nxt[0]
        order.append(cur)
        vis.add(cur)
    if len(order) < npts:
        order.extend([i for i in range(npts) if i not in vis])
    return order


def boundary_loop_order(npts: int, elements: np.ndarray) -> list[int]:
    edge_count: dict[tuple[int, int], int] = {}
    for elem in elements:
        for a, b in zip(elem, np.roll(elem, -1)):
            a = int(a)
            b = int(b)
            e = (min(a, b), max(a, b))
            edge_count[e] = edge_count.get(e, 0) + 1
    boundary = [e for e, c in edge_count.items() if c == 1]
    if not boundary:
        return []
    adj = {i: [] for i in range(npts)}
    bnodes = set()
    for a, b in boundary:
        adj[a].append(b)
        adj[b].append(a)
        bnodes.add(a)
        bnodes.add(b)
    endpoints = [i for i in bnodes if len(adj[i]) == 1]
    start = endpoints[0] if endpoints else min(bnodes)
    order = [start]
    vis = {start}
    cur = start
    prev = None
    while True:
        nbrs = adj[cur]
        nxts = [n for n in nbrs if n != prev]
        if not nxts:
            break
        nxt = nxts[0]
        if nxt in vis:
            if nxt == order[0] and len(vis) == len(bnodes):
                break
            alt = [n for n in nbrs if n not in (prev, nxt)]
            if alt:
                nxt = alt[0]
            else:
                break
        order.append(nxt)
        vis.add(nxt)
        prev, cur = cur, nxt
        if len(vis) >= len(bnodes) and order[0] in adj[cur]:
            break
    return [i for i in order if i in bnodes]


def nearest_neighbor_order(xy: np.ndarray) -> list[int]:
    n = xy.shape[0]
    if n == 0:
        return []
    unused = set(range(n))
    start = int(np.argmin(xy[:, 0]))
    order = [start]
    unused.remove(start)
    last = start
    while unused:
        idxs = np.array(sorted(list(unused)))
        d = np.linalg.norm(xy[idxs] - xy[last], axis=1)
        j = idxs[int(np.argmin(d))]
        order.append(int(j))
        unused.remove(int(j))
        last = int(j)
    return order


def order_zone(z: SimpleNamespace, x_idx: int, y_idx: int) -> np.ndarray:
    X = z.nodes[:, x_idx]
    Y = z.nodes[:, y_idx]
    if z.elem is not None:
        if z.elem.shape[1] == 2:
            ord_idx = order_points_from_lineseg(len(X), z.elem)
        else:
            ord_idx = boundary_loop_order(len(X), z.elem)
            if not ord_idx:
                ord_idx = nearest_neighbor_order(np.column_stack([X, Y]))
    else:
        ord_idx = nearest_neighbor_order(np.column_stack([X, Y]))
    return np.array(ord_idx, dtype=int)


def read_solution(path: Path):
    with open(path, "r") as f:
        lines = f.readlines()

    var_line = next((ln for ln in lines if ln.lstrip().upper().startswith("VARIABLES")), "")
    var_names = re.findall(r'"([^\"]+)"', var_line)
    var_map = {_normalize(name): i for i, name in enumerate(var_names)}
    n_vars = len(var_names)

    z_idx = _get_var_index(var_map, ["z"])

    zone_starts = [i for i, line in enumerate(lines) if line.lstrip().startswith("ZONE")]
    zone_starts.append(len(lines))

    # Discover wall zones dynamically by inspecting each zone header/title for
    # a set of keywords (see WALL_KEYWORDS above).  The resulting list of
    # indices is used throughout the parsing below.
    wall_zone_indices: list[int] = []
    for idx, start in enumerate(zone_starts[:-1], start=1):
        header = lines[start]
        title_match = re.search(r'T="([^"]+)', header)
        title = title_match.group(1) if title_match else ""
        meta = f"{header} {title}".lower()
        if any(kw in meta for kw in WALL_KEYWORDS):
            wall_zone_indices.append(idx)

    wall_zone_set = set(wall_zone_indices)

    wall_zones: list[SimpleNamespace] = []
    total_nodes = 0
    wall_nodes = 0

    for idx, (start, end) in enumerate(zip(zone_starts, zone_starts[1:]), start=1):
        header = lines[start]
        mN = re.search(r"N=\s*(\d+)", header)
        if not mN:
            continue
        N = int(mN.group(1))
        total_nodes += N
        if idx not in wall_zone_set:
            continue
        mE = re.search(r"E=\s*(\d+)", header)
        E = int(mE.group(1)) if mE else 0
        mtype = re.search(r"ZONETYPE=([^,\s]+)", header)
        zonetype = mtype.group(1).upper() if mtype else ""

        nnpe = expected_nodes_per_element(zonetype)
        data_lines = lines[start + 1 : end]
        text = " ".join(line.strip() for line in data_lines)
        # Some Tecplot exports omit the "E" in scientific notation (e.g.
        # "1.23+05").  Insert the missing "e" so NumPy can parse the values.
        text = re.sub(r"(?<=\d)([+-]\d{2,})", r"e\1", text)
        values = np.fromstring(text, sep=" ")
        node_vals = values[: N * n_vars].reshape(N, n_vars)
        if nnpe > 0 and E > 0 and values.size >= N * n_vars + E * nnpe:
            conn_vals = values[N * n_vars : N * n_vars + E * nnpe].reshape(E, nnpe).astype(int) - 1
        else:
            conn_vals = None

        mask = node_vals[:, z_idx] <= 0
        nodes = node_vals[mask]
        if conn_vals is not None:
            idx_map = {old: new for new, old in enumerate(np.where(mask)[0])}
            new_elems = []
            for elem in conn_vals:
                elem = [int(n) for n in elem]
                if all(mask[n] for n in elem):
                    new_elems.append([idx_map[n] for n in elem])
            elem_arr = np.array(new_elems, dtype=int)
        else:
            elem_arr = None
        wall_nodes += nodes.shape[0]
        wall_zones.append(SimpleNamespace(nodes=nodes, elem=elem_arr))

    return wall_zones, total_nodes, wall_nodes, var_map, wall_zone_indices


def write_tecplot(path: Path, x: np.ndarray, y: np.ndarray, cp: np.ndarray):
    with open(path, "w") as f:
        f.write('TITLE = "Merged Wall Cp"\n')
        f.write('VARIABLES = "X" "Y" "Cp"\n')
        f.write(f'ZONE T="MergedWall", I={len(x)}, DATAPACKING=POINT\n')
        for xi, yi, ci in zip(x, y, cp):
            f.write(f"{xi} {yi} {ci}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Extract wall nodes and compute surface pressure coefficient."
    )
    parser.add_argument("solution", type=Path, help="Path to the solution .dat file")
    parser.add_argument("--out", type=Path, help="Optional Tecplot output file")
    args = parser.parse_args()

    wall_zones, total_nodes, wall_nodes, var_map, wall_zone_indices = read_solution(args.solution)
    print(
        f"Total nodes: {total_nodes}, wall nodes: {wall_nodes}, "
        f"excluded: {total_nodes - wall_nodes}"
    )
    print(f"Detected wall zone indices: {wall_zone_indices}")

    if not wall_zones:
        return

    idx = lambda *names: _get_var_index(var_map, list(names))
    x_idx = idx("x")
    y_idx = idx("y")
    p_idx = idx("pressure", "p")
    rho_idx = idx("density", "rho")
    u_idx = idx("u", "velocityx", "xvelocity")
    v_idx = idx("v", "velocityy", "yvelocity")
    w_idx = idx("w", "velocityz", "zvelocity")

    first = wall_zones[0].nodes[0]
    rho_inf = first[rho_idx]
    p_inf = first[p_idx]
    u_inf = float(np.sqrt(first[u_idx] ** 2 + first[v_idx] ** 2 + first[w_idx] ** 2))

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    cps: list[np.ndarray] = []

    for z in wall_zones:
        ord_idx = order_zone(z, x_idx, y_idx)
        nodes = z.nodes[ord_idx]
        x = nodes[:, x_idx]
        y = nodes[:, y_idx]
        p = nodes[:, p_idx]
        cp = (p - p_inf) / (0.5 * rho_inf * u_inf ** 2)
        xs.append(x)
        ys.append(y)
        cps.append(cp)

    x = np.concatenate(xs)
    y = np.concatenate(ys)
    cp = np.concatenate(cps)

    fig1, ax1 = plt.subplots()
    ax1.scatter(x, y, s=5)
    ax1.set_aspect("equal", adjustable="box")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_title("Airfoil geometry (z<=0)")
    fig1.savefig("airfoil_geometry.png")

    x_closed = np.append(x, x[0])
    y_closed = np.append(y, y[0])
    cp_closed = np.append(cp, cp[0])

    fig2, ax2 = plt.subplots()
    ax2.plot(x_closed, cp_closed)
    ax2.set_xlabel("x")
    ax2.set_ylabel("Cp")
    ax2.invert_yaxis()
    ax2.set_title("Surface Cp")
    fig2.savefig("surface_cp.png")

    if args.out:
        write_tecplot(args.out, x_closed, y_closed, cp_closed)

    try:
        plt.show()
    except Exception:
        pass


if __name__ == "__main__":
    main()

