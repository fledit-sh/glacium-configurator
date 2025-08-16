import re
from pathlib import Path
import argparse
from types import SimpleNamespace
from typing import Optional
import tempfile
import zipfile
from scipy.interpolate import CubicSpline

import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import scienceplots
plt.style.use("science")


# Keywords used to automatically identify wall zones.  The search is case
# insensitive and runs against the zone title as well as the raw zone header.
# Additional markers can be added to this list as needed.
WALL_KEYWORDS = ("wall", "surface", "solid")

# Zonetype values that represent surface elements. For these zones the
# element connectivity describes faces and must be reduced to boundary edges
# before further processing.
SURFACE_ZONETYPES = {"FEQUADRILATERAL", "FETRIANGLE"}


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


def walk_zone_nodes(z: SimpleNamespace) -> tuple[np.ndarray, int, bool]:
    """Return ordered node indices, endpoint count and closed-loop flag.

    The element connectivity may describe multiple disconnected segments.  All
    segments are walked in a deterministic order by computing connected
    components and traversing each component sequentially.  Nodes that do not
    participate in any element are included at the end in their original index
    order.  The second value returned is the number of nodes with degree one
    ("endpoints").  If all connected nodes have degree two the third value is
    ``True`` indicating the component(s) form a closed loop.
    """

    n_nodes = len(z.nodes)
    if z.elem is None or z.elem.ndim != 2 or z.elem.size == 0:
        return np.arange(n_nodes, dtype=int), 0, False

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

    degree1_count = sum(1 for neighbors in adj.values() if len(neighbors) == 1)
    all_degree2 = all(len(neighbors) == 2 for neighbors in adj.values()) and bool(adj)
    is_closed = degree1_count == 0 and all_degree2

    visited_nodes: set[int] = set()
    order: list[int] = []

    for node in range(n_nodes):
        if node in visited_nodes:
            continue

        if node not in adj:
            # Isolated node with no connectivity.
            order.append(node)
            visited_nodes.add(node)
            continue

        # Discover the full connected component for this node.
        stack = [node]
        component: set[int] = {node}
        visited_nodes.add(node)
        while stack:
            cur = stack.pop()
            for nb in adj[cur]:
                if nb not in visited_nodes:
                    visited_nodes.add(nb)
                    component.add(nb)
                    stack.append(nb)

        # Choose a starting node: prefer an endpoint if present for deterministic
        # traversal of open paths; otherwise use the smallest index.
        endpoint_candidates = [n for n in component if len(adj[n]) == 1]
        start = min(endpoint_candidates) if endpoint_candidates else min(component)

        comp_order = [start]
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
            comp_order.append(next_node)
            current = next_node

        order.extend(comp_order)

    return np.array(order, dtype=int), degree1_count, is_closed


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


def slice_surface_zone(
    nodes: np.ndarray, conn_vals: np.ndarray, z_threshold: float, z_idx: int
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Slice surface elements by a plane ``z=z_threshold``.

    Parameters
    ----------
    nodes : np.ndarray
        Array of node values for the entire zone.
    conn_vals : np.ndarray
        Element connectivity describing faces (triangles or quadrilaterals).
    z_threshold : float
        ``z`` value of the slicing plane.
    z_idx : int
        Column index of the ``z`` coordinate in ``nodes``.

    Returns
    -------
    tuple of np.ndarray and Optional[np.ndarray]
        The sliced node array containing newly created intersection points and
        an array of boundary edges referencing those nodes. If no intersections
        are found, the edge array is ``None``.
    """

    if conn_vals is None or conn_vals.size == 0:
        return np.empty((0, nodes.shape[1])), None

    # Maps for de-duplicating nodes
    node_map: dict[tuple[str, int], int] = {}
    new_nodes: list[np.ndarray] = []
    edges: list[tuple[int, int]] = []

    def _add_existing(idx: int) -> int:
        key = ("n", idx)
        if key not in node_map:
            node_map[key] = len(new_nodes)
            new_nodes.append(nodes[idx])
        return node_map[key]

    def _add_intersection(a: int, b: int) -> int:
        key = ("i",) + tuple(sorted((a, b)))
        if key not in node_map:
            za, zb = nodes[a, z_idx], nodes[b, z_idx]
            t = (z_threshold - za) / (zb - za)
            point = nodes[a] + t * (nodes[b] - nodes[a])
            node_map[key] = len(new_nodes)
            new_nodes.append(point)
        return node_map[key]

    for elem in conn_vals:
        verts = [int(n) for n in elem]
        ints: list[int] = []
        n = len(verts)
        for i in range(n):
            a = verts[i]
            b = verts[(i + 1) % n]
            za = nodes[a, z_idx] - z_threshold
            zb = nodes[b, z_idx] - z_threshold
            if za == 0 and zb == 0:
                ia = _add_existing(a)
                ib = _add_existing(b)
                edges.append((ia, ib))
            elif za == 0:
                ia = _add_existing(a)
                ints.append(ia)
            elif zb == 0:
                ib = _add_existing(b)
                ints.append(ib)
            elif za * zb < 0:
                ints.append(_add_intersection(a, b))
        if len(ints) == 2:
            edges.append((ints[0], ints[1]))
        elif len(ints) > 2:
            ints = list(dict.fromkeys(ints))
            for i in range(len(ints) - 1):
                edges.append((ints[i], ints[i + 1]))

    # NEU: Undirected-Dedupe der Kanten
    if edges:
        uniq = sorted({tuple(sorted(e)) for e in edges})
        edge_arr = np.array(uniq, dtype=int)
    else:
        edge_arr = None

    return np.array(new_nodes), edge_arr


def read_solution(path: Path, z_threshold: float = 0.0, tol: float = 0.0):
    """Parse a Tecplot solution file and extract wall and inlet zones.

    Wall zones are filtered so that only nodes with ``z <= z_threshold + tol``
    are included. Element connectivity is rebuilt to reference the filtered
    nodes, ensuring downstream code receives zones that already honor the
    filtering mask.
    """

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
            elem_arr = None
            if conn_vals is not None:
                idx_map = {old: new for new, old in enumerate(np.where(mask)[0])}
                new_elems: list[list[int]] = []
                for elem in conn_vals:
                    elem = [int(n) for n in elem]
                    if all(mask[n] for n in elem):
                        new_elems.append([idx_map[n] for n in elem])
                if new_elems:
                    if zonetype in SURFACE_ZONETYPES:
                        edge_counts: dict[tuple[int, int], int] = defaultdict(int)
                        for elem in new_elems:
                            for a, b in zip(elem, elem[1:]):
                                edge = tuple(sorted((a, b)))
                                edge_counts[edge] += 1
                        boundary_edges = [
                            edge
                            for edge, count in edge_counts.items()
                            if count == 1 and edge[0] != edge[1]
                        ]
                        if boundary_edges:
                            elem_arr = np.array(boundary_edges, dtype=int)
                    else:
                        elem_arr = np.array(new_elems, dtype=int)
                if elem_arr is None and zonetype in SURFACE_ZONETYPES:
                    nodes, elem_arr = slice_surface_zone(
                        node_vals, conn_vals, z_threshold + tol, z_idx
                    )
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
        var_names,
        wall_zone_indices,
        inlet_zone_indices,
    )


def write_tecplot(path: Path, nodes: np.ndarray, conn: np.ndarray, var_names: list[str]):
    """Write merged wall data to a Tecplot ASCII file.

    Parameters
    ----------
    path : Path
        Output file location.
    nodes : np.ndarray
        Node data where the last column contains ``Cp`` values and the remaining
        columns correspond to ``var_names``.
    conn : np.ndarray
        Element connectivity using zero-based node indices.
    var_names : list[str]
        Names for the columns in ``nodes`` excluding the final ``Cp`` column.
    """

    var_line = " ".join(f'"{v}"' for v in var_names + ["Cp"])
    n_nodes = int(nodes.shape[0])
    n_elem = int(conn.shape[0]) if conn is not None else 0

    nodes_with_cp = nodes

    with open(path, "w") as f:
        f.write('TITLE = "Merged Wall Cp"\n')
        f.write(f'VARIABLES = {var_line}\n')
        f.write(
            f'ZONE T="MergedWall", N={n_nodes}, E={n_elem}, '  # type: ignore[fmt]
            'DATAPACKING=POINT, ZONETYPE=FELINESEG\n'
        )
        for row in nodes_with_cp:
            f.write(" ".join(str(v) for v in row) + "\n")
        if conn is not None:
            for a, b in conn:
                f.write(f"{a + 1} {b + 1}\n")

def plot_cp_normals_outward(x, y, cp, scale=0.05):
    # Doppelte Punkte entfernen
    coords = np.column_stack((x, y))
    mask = np.ones(len(coords), dtype=bool)
    mask[1:] = np.any(np.diff(coords, axis=0) != 0, axis=1)
    x = x[mask]
    y = y[mask]
    cp = cp[mask]
    # Schritt 1: Orientierung bestimmen (signed area)
    area = 0.5 * np.sum(x[:-1]*y[1:] - x[1:]*y[:-1])
    ccw = area > 0  # True = gegen Uhrzeigersinn

    # Schritt 2: Tangenten
    dx = np.gradient(x)
    dy = np.gradient(y)

    # Schritt 3: Normale immer nach außen
    if ccw:
        nx = dy
        ny = -dx
    else:
        nx = -dy
        ny = dx

    # Normieren
    norm = np.sqrt(nx**2 + ny**2)
    nx /= norm
    ny /= norm

    # Schritt 4: Länge = |Cp| * scale
    lengths = np.abs(cp) * scale

    # Schritt 5: Farbe nach Vorzeichen von Cp
    colors = ['red' if c > 0 else 'blue' for c in cp]

    # Schritt 6: Plot
    fig, ax = plt.subplots()
    ax.plot(x, y, 'k-', lw=1)
    for xi, yi, nxi, nyi, length, color in zip(x, y, nx, ny, lengths, colors):
        ax.plot([xi, xi + nxi * length],
                [yi, yi + nyi * length],
                color=color, lw=1)

    ax.set_aspect('equal', adjustable='box')
    ax.set_title("Cp-Normalen (immer nach außen)")
    plt.show()

# Beispiel:
# plot_cp_normals_outward(x_closed, y_closed, cp_closed)


def merge_zones(
    wall_zones: list[SimpleNamespace],
    inlet_zones: list[SimpleNamespace],
    var_map: dict[str, int],
    return_full: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray]:
    """Merge ordered wall zones and compute the closed Cp curve.

    Parameters
    ----------
    wall_zones : list of SimpleNamespace
        Zones containing ``nodes`` and optional ``elem`` arrays.
        The ``nodes`` arrays must already be filtered so that only points
        satisfying the ``z``-threshold remain; connectivity in ``elem`` should
        reflect the filtered nodes. No additional ``z`` filtering is performed.
    inlet_zones : list of SimpleNamespace
        Zones used to estimate free–stream conditions. May be empty.
    var_map : dict
        Mapping from normalized variable names to column indices.

    Returns
    -------
    tuple of np.ndarray
        If ``return_full`` is ``False`` (default), returns the closed ``x``,
        ``y`` and ``Cp`` arrays describing the airfoil surface.  When
        ``return_full`` is ``True`` the full node array with an appended ``Cp``
        column and the corresponding connectivity are returned instead.
    """

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
    for z_idx, z in enumerate(wall_zones, start=1):
        z_title = getattr(z, "title", "")
        # ``read_solution`` filters each zone by ``z`` and remaps its
        # connectivity.  The merged edges below therefore operate directly on
        # these filtered nodes.
        local_order, n_endpoints, is_closed = walk_zone_nodes(z)
        if is_closed:
            raise ValueError(
                "Zone forms a closed loop; "
                "run a boundary-extraction step to obtain open boundary edges"
            )
        if n_endpoints == 0:
            title_info = f" ({z_title})" if z_title else ""
            raise ValueError(
                f"Zone {z_idx}{title_info} has no endpoints; "
                "run a boundary-extraction step to generate boundary edges"
            )
        if n_endpoints != 2:
            raise ValueError(
                f"Zone has {n_endpoints} endpoints; expected 2"
            )
        ordered_nodes = z.nodes[local_order]
        # Für die erste Zone keine Orientierung erzwingen.
        # Für nachfolgende Zonen: so orientieren, dass der erste Punkt
        # am nächsten zum Endpunkt der vorherigen Zone liegt.

        if prev_end is not None and len(nodes_list) > 0:
            prev_pt = nodes_list[-1][-1, [x_idx, y_idx]]
            d_start = np.linalg.norm(ordered_nodes[0, [x_idx, y_idx]] - prev_pt)
            d_end = np.linalg.norm(ordered_nodes[-1, [x_idx, y_idx]] - prev_pt)
            if d_end < d_start:
                ordered_nodes = ordered_nodes[::-1]

        n = ordered_nodes.shape[0]
        start_global = offset
        end_global = offset + n - 1
        z.start = start_global
        z.end = end_global


        nodes_list.append(ordered_nodes)
        elem_list.append(
            np.column_stack(
                [
                    np.arange(start_global, start_global + n - 1),
                    np.arange(start_global + 1, start_global + n),
                ]
            )
        )
        if prev_end is not None:
            elem_list.append(np.array([[prev_end, start_global]], dtype=int))
        offset += n
        prev_end = end_global

    all_nodes = np.concatenate(nodes_list)
    if prev_end is not None and all_nodes.size:
        elem_list.append(np.array([[prev_end, wall_zones[0].start]], dtype=int))
    all_elem = np.concatenate(elem_list) if elem_list else None

    merged_zone = SimpleNamespace(nodes=all_nodes, elem=all_elem)
    ord_idx, _, _ = walk_zone_nodes(merged_zone)
    nodes = all_nodes[ord_idx]

    x = nodes[:, x_idx]
    y = nodes[:, y_idx]
    p = nodes[:, p_idx]
    cp = (p - p_inf) / (0.5 * rho_inf * u_inf ** 2)

    # Validate Cp continuity around the loop. Only flag extreme jumps which
    # typically indicate corrupted data rather than expected pressure
    # differences between surfaces.
    if cp.size > 1:
        cp_diff = np.abs(np.diff(np.append(cp, cp[0])))
        if float(cp_diff.max()) > 10.0:
            raise ValueError("Cp curve exhibits a discontinuity at the closing point")

    nodes_cp = np.column_stack([nodes, cp])

    if all_elem is None:
        n_conn = nodes_cp.shape[0]
        conn_ordered = np.column_stack(
            [np.arange(n_conn), np.roll(np.arange(n_conn), -1)]
        )
    else:
        rev_idx = np.empty_like(ord_idx)
        rev_idx[ord_idx] = np.arange(len(ord_idx))
        conn_ordered = rev_idx[all_elem]
        closing = np.array([[nodes_cp.shape[0] - 1, 0]], dtype=int)
        if not np.any(np.all(conn_ordered == closing, axis=1)):
            conn_ordered = np.vstack([conn_ordered, closing])

    x_closed = np.append(x, x[0])
    y_closed = np.append(y, y[0])
    cp_closed = np.append(cp, cp[0])

    if return_full:
        return nodes_cp, conn_ordered

    return x_closed, y_closed, cp_closed


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
            var_names,
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

        if args.out:
            nodes_cp, conn_ordered = merge_zones(
                wall_zones, inlet_zones, var_map, return_full=True
            )
            x_idx = _get_var_index(var_map, ["x"])
            y_idx = _get_var_index(var_map, ["y"])
            x = nodes_cp[:, x_idx]
            y = nodes_cp[:, y_idx]
            cp = nodes_cp[:, -1]
            x_closed = np.append(x, x[0])
            y_closed = np.append(y, y[0])
            cp_closed = np.append(cp, cp[0])
        else:
            x_closed, y_closed, cp_closed = merge_zones(
                wall_zones, inlet_zones, var_map
            )

        # bestehende Plots
        geom_path = out_dir / f"{prefix}_airfoil_geometry.png"
        cp_path = out_dir / f"{prefix}_surface_cp.png"
        plot_airfoil_geometry(x_closed, y_closed, geom_path)
        plot_surface_cp(x_closed, cp_closed, cp_path)

        # NEU: Cp-Normalen-Plot
        plot_cp_normals_outward(x_closed, y_closed, cp_closed, scale=0.05)

        geom_path = out_dir / f"{prefix}_airfoil_geometry.png"
        cp_path = out_dir / f"{prefix}_surface_cp.png"
        plot_airfoil_geometry(x_closed, y_closed, geom_path)
        plot_surface_cp(x_closed, cp_closed, cp_path)
        if args.out:
            write_tecplot(args.out, nodes_cp, conn_ordered, var_names)

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

