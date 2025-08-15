import re
from pathlib import Path
import argparse
from types import SimpleNamespace
from typing import Optional
import tempfile
import zipfile

import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict


# Keywords used to automatically identify wall zones.  The search is case
# insensitive and runs against the zone title as well as the raw zone header.
# Additional markers can be added to this list as needed.
WALL_KEYWORDS = ("wall", "surface", "solid")


def _normalize(name: str) -> str:
    """Normalize a Tecplot variable name for lookup.

    The Tecplot exports used with this script often append units or other
    annotations to the variable names, e.g. ``"Pressure (N/m^2)"`` or
    ``"Cfx; Velocity"``.  For lookup purposes we only care about the base
    name, so everything after the first space, ``(`` or ``;`` is discarded and
    any remaining punctuation is stripped before lowering the case.
    """
    name = name.strip()
    # Truncate at the first space, '(' or ';'.
    name = re.split(r"[\s(;]", name, 1)[0]
    # Remove all punctuation/special characters.
    name = re.sub(r"[^A-Za-z0-9]", "", name)
    return name.lower()


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


def walk_zone_nodes(z: SimpleNamespace) -> np.ndarray:
    """Return node indices by walking element connectivity sequentially."""

    n_nodes = len(z.nodes)
    if z.elem is None or z.elem.ndim != 2 or z.elem.size == 0:
        return np.arange(n_nodes, dtype=int)

    # Build adjacency from element connectivity.  Each element contributes
    # edges between successive nodes and is closed if it has more than two
    # vertices.
    adj: dict[int, list[int]] = defaultdict(list)
    for elem in z.elem:
        nodes = [int(n) for n in elem]
        for a, b in zip(nodes, nodes[1:]):
            adj[a].append(b)
            adj[b].append(a)
        if len(nodes) > 2:
            adj[nodes[0]].append(nodes[-1])
            adj[nodes[-1]].append(nodes[0])

    start = int(z.elem[0][0])
    order = [start]
    current = start
    visited_edges: set[tuple[int, int]] = set()
    while True:
        neighbors = adj[current]
        next_node = None
        for n in neighbors:
            edge = tuple(sorted((current, n)))
            if edge not in visited_edges:
                visited_edges.add(edge)
                next_node = n
                break
        if next_node is None or next_node == start:
            break
        order.append(next_node)
        current = next_node

    return np.array(order, dtype=int)


def filter_nodes_by_z(
    nodes: np.ndarray, elem: Optional[np.ndarray], z_idx: int
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    mask = nodes[:, z_idx] <= 0
    filtered_nodes = nodes[mask]
    if elem is None:
        return filtered_nodes, None
    idx_map = {old: new for new, old in enumerate(np.where(mask)[0])}
    new_elems: list[list[int]] = []
    for a, b in elem:
        if mask[a] and mask[b]:
            new_elems.append([idx_map[a], idx_map[b]])
    filtered_elem = np.array(new_elems, dtype=int) if new_elems else None
    return filtered_nodes, filtered_elem


def read_solution(path: Path, z_threshold: float = 0.0, tol: float = 0.0):
    with open(path, "r") as f:
        lines = f.readlines()

    var_line = next((ln for ln in lines if ln.lstrip().upper().startswith("VARIABLES")), "")
    var_names = re.findall(r'"([^\"]+)"', var_line)
    var_map = {_normalize(name): i for i, name in enumerate(var_names)}
    n_vars = len(var_names)

    z_idx = _get_var_index(var_map, ["z"])

    zone_starts = [i for i, line in enumerate(lines) if line.lstrip().startswith("ZONE")]
    zone_starts.append(len(lines))

    # Discover wall and inlet zones dynamically by inspecting each zone header
    # and title. The search is case insensitive and runs against the zone title
    # as well as the raw zone header.
    wall_zone_indices: list[int] = []
    inlet_zone_indices: list[int] = []
    for idx, start in enumerate(zone_starts[:-1], start=1):
        header = lines[start]
        title_match = re.search(r'T="([^\"]+)', header)
        title = title_match.group(1) if title_match else ""
        meta = f"{header} {title}".lower()
        if any(kw in meta for kw in WALL_KEYWORDS):
            wall_zone_indices.append(idx)
        if "inlet" in meta:
            inlet_zone_indices.append(idx)

    wall_zone_set = set(wall_zone_indices)
    inlet_zone_set = set(inlet_zone_indices)

    wall_zones: list[SimpleNamespace] = []
    inlet_zones: list[SimpleNamespace] = []
    total_nodes = 0
    wall_nodes = 0

    for idx, (start, end) in enumerate(zip(zone_starts, zone_starts[1:]), start=1):
        header = lines[start]
        mN = re.search(r"N=\s*(\d+)", header)
        if not mN:
            continue
        N = int(mN.group(1))
        total_nodes += N

        is_wall = idx in wall_zone_set
        is_inlet = idx in inlet_zone_set
        if not (is_wall or is_inlet):
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

        if is_wall:
            mask = node_vals[:, z_idx] <= z_threshold + tol
            nodes = node_vals[mask]
            if conn_vals is not None:
                idx_map = {old: new for new, old in enumerate(np.where(mask)[0])}
                new_elems = []
                for elem in conn_vals:
                    elem = [int(n) for n in elem]
                    if all(mask[n] for n in elem):
                        new_elems.append([idx_map[n] for n in elem])
                # Ensure downstream code receives either a 2-D array or None.
                elem_arr = (
                    np.array(new_elems, dtype=int) if new_elems else None
                )
            else:
                elem_arr = None
            wall_nodes += nodes.shape[0]
            wall_zones.append(SimpleNamespace(nodes=nodes, elem=elem_arr))

        if is_inlet:
            inlet_zones.append(SimpleNamespace(nodes=node_vals))

    return (
        wall_zones,
        inlet_zones,
        total_nodes,
        wall_nodes,
        var_map,
        wall_zone_indices,
        inlet_zone_indices,
    )


def write_tecplot(path: Path, x: np.ndarray, y: np.ndarray, cp: np.ndarray):
    with open(path, "w") as f:
        f.write('TITLE = "Merged Wall Cp"\n')
        f.write('VARIABLES = "X" "Y" "Cp"\n')
        f.write(f'ZONE T="MergedWall", I={len(x)}, DATAPACKING=POINT\n')
        for xi, yi, ci in zip(x, y, cp):
            f.write(f"{xi} {yi} {ci}\n")


def plot_airfoil_geometry(x: np.ndarray, y: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots()
    ax.scatter(x, y, s=5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Airfoil geometry (z<=0)")
    fig.savefig(path)
    plt.close(fig)


def plot_surface_cp(x: np.ndarray, cp: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots()
    ax.plot(x, cp)
    ax.set_xlabel("x")
    ax.set_ylabel("Cp")
    ax.invert_yaxis()
    ax.set_title("Surface Cp")
    fig.savefig(path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Extract wall nodes and compute surface pressure coefficient."
    )
    parser.add_argument("solution", type=Path, help="Path to the solution .dat file")
    parser.add_argument("--out", type=Path, help="Optional Tecplot output file")
    parser.add_argument(
        "--z-threshold",
        type=float,
        default=0.0,
        help="Z cutoff for wall extraction (uses Z <= threshold + tol)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.0,
        help="Additional tolerance added to z-threshold",
    )
    args = parser.parse_args()

    prefix = args.solution.stem
    out_dir = args.out.parent if args.out else args.solution.parent

    def process(sol_path: Path) -> None:
        (
            wall_zones,
            inlet_zones,
            total_nodes,
            wall_nodes,
            var_map,
            wall_zone_indices,
            inlet_zone_indices,
        ) = read_solution(sol_path, args.z_threshold, args.tolerance)
        print(
            f"Total nodes: {total_nodes}, wall nodes: {wall_nodes}, "
            f"excluded: {total_nodes - wall_nodes}"
        )
        print(f"Detected wall zone indices: {wall_zone_indices}")
        print(f"Detected inlet zone indices: {inlet_zone_indices}")

        if not wall_zones:
            return

        idx = lambda *names: _get_var_index(var_map, list(names))
        x_idx = idx("x")
        y_idx = idx("y")
        p_idx = idx("pressure", "p")
        rho_idx = idx("density", "rho")
        u_idx = idx("u", "velocityx", "xvelocity", "v1", "v1-velocity")
        v_idx = idx("v", "velocityy", "yvelocity", "v2", "v2-velocity")
        w_idx = idx("w", "velocityz", "zvelocity", "v3", "v3-velocity")
        z_idx = idx("z")

        if inlet_zones:
            inlet_nodes = np.concatenate([z.nodes for z in inlet_zones])
            rho_inf = float(np.median(inlet_nodes[:, rho_idx]))
            p_inf = float(np.median(inlet_nodes[:, p_idx]))
            vel_mag = np.sqrt(
                inlet_nodes[:, u_idx] ** 2
                + inlet_nodes[:, v_idx] ** 2
                + inlet_nodes[:, w_idx] ** 2
            )
            u_inf = float(np.median(vel_mag))
        else:
            first = wall_zones[0].nodes[0]
            rho_inf = first[rho_idx]
            p_inf = first[p_idx]
            u_inf = float(
                np.sqrt(first[u_idx] ** 2 + first[v_idx] ** 2 + first[w_idx] ** 2)
            )

        nodes_list: list[np.ndarray] = []
        elem_list: list[np.ndarray] = []
        offset = 0
        prev_end = None
        for z in wall_zones:
            local_order = walk_zone_nodes(z)
            ordered_nodes = z.nodes[local_order]
            n = ordered_nodes.shape[0]
            start_global = offset
            end_global = offset + n - 1
            z.first = start_global
            z.last = end_global
            nodes_list.append(ordered_nodes)
            elem_list.append(
                np.column_stack([
                    np.arange(start_global, start_global + n - 1),
                    np.arange(start_global + 1, start_global + n),
                ])
            )
            if prev_end is not None:
                elem_list.append(np.array([[prev_end, start_global]], dtype=int))
            offset += n
            prev_end = end_global

        all_nodes = np.concatenate(nodes_list)
        # Connect final zone back to the first to close the loop
        if prev_end is not None and all_nodes.size:
            elem_list.append(np.array([[prev_end, 0]], dtype=int))
        all_elem = np.concatenate(elem_list) if elem_list else None

        all_nodes, all_elem = filter_nodes_by_z(all_nodes, all_elem, z_idx)

        merged_zone = SimpleNamespace(nodes=all_nodes, elem=all_elem)
        ord_idx = walk_zone_nodes(merged_zone)
        nodes = all_nodes[ord_idx]

        x = nodes[:, x_idx]
        y = nodes[:, y_idx]
        p = nodes[:, p_idx]
        cp = (p - p_inf) / (0.5 * rho_inf * u_inf ** 2)

        x_closed = np.append(x, x[0])
        y_closed = np.append(y, y[0])
        cp_closed = np.append(cp, cp[0])

        geom_path = out_dir / f"{prefix}_airfoil_geometry.png"
        cp_path = out_dir / f"{prefix}_surface_cp.png"
        plot_airfoil_geometry(x_closed, y_closed, geom_path)
        plot_surface_cp(x_closed, cp_closed, cp_path)

        if args.out:
            write_tecplot(args.out, x_closed, y_closed, cp_closed)

    sol_path = args.solution
    if sol_path.suffix.lower() == ".zip":
        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(sol_path, "r") as zf:
                zf.extractall(tmpdir)
            dat_files = list(Path(tmpdir).rglob("*.dat"))
            if not dat_files:
                raise FileNotFoundError("No .dat file found in archive")
            process(dat_files[0])
    else:
        process(sol_path)


if __name__ == "__main__":
    main()

