import re
from pathlib import Path
import argparse

import numpy as np
import matplotlib.pyplot as plt


def read_solution(path: Path):
    """Read solution file and return wall node data.

    The Tecplot solution file may contain several zones.  Wall data are stored
    either in zones whose titles contain the word ``wall`` or in known index
    positions.  This function iterates over every zone in the file, extracts the
    node data for the wall zones, concatenates them and returns the resulting
    array.
    """

    with open(path, "r") as f:
        lines = f.readlines()

    # Locate all ZONE headers
    zone_starts = [i for i, line in enumerate(lines) if line.lstrip().startswith("ZONE")]
    zone_starts.append(len(lines))  # sentinel to simplify slicing

    wall_data: list[np.ndarray] = []
    n_vars = 21  # number of variables per node

    # Known wall zone indices if titles are not descriptive
    known_wall_indices = {2, 3}

    for idx, (start, end) in enumerate(zip(zone_starts, zone_starts[1:]), start=1):
        header = lines[start]

        # Extract node count from the header; skip if not present
        match = re.search(r"N=\s*(\d+)", header)
        if not match:
            continue
        n_nodes = int(match.group(1))

        # Extract zone title, if any
        title_match = re.search(r'T="([^"]+)"', header)
        title = title_match.group(1) if title_match else ""

        # Concatenate all data lines for this zone and parse numbers
        data_lines = lines[start + 1 : end]
        text = "".join(line.rstrip("\n") for line in data_lines)
        values = np.fromstring(text, sep=" ")

        # Reshape into (n_nodes, n_vars) and discard any extra values (e.g. connectivity)
        zone_data = values[: n_nodes * n_vars].reshape(n_nodes, n_vars)

        if "wall" in title.lower() or idx in known_wall_indices:
            wall_data.append(zone_data)

    if not wall_data:
        return np.empty((0, n_vars))

    return np.concatenate(wall_data, axis=0)

def write_tecplot(path: Path, x: np.ndarray, y: np.ndarray, cp: np.ndarray):
    """Write arrays to Tecplot ASCII file."""
    with open(path, 'w') as f:
        f.write('TITLE = "Merged Wall Cp"\n')
        f.write('VARIABLES = "X" "Y" "Cp"\n')
        f.write(f'ZONE T="MergedWall", I={len(x)}, DATAPACKING=POINT\n')
        for xi, yi, ci in zip(x, y, cp):
            f.write(f"{xi} {yi} {ci}\n")


def nearest_neighbor_order(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return an index order tracing the boundary using nearest-neighbor traversal."""
    points = np.column_stack((x, y))
    n = len(points)
    if n == 0:
        return np.array([], dtype=int)

    order = [0]
    remaining = set(range(1, n))
    current = 0

    while remaining:
        last = points[current]
        rem_list = np.array(list(remaining))
        distances = np.linalg.norm(points[rem_list] - last, axis=1)
        next_idx = rem_list[np.argmin(distances)]
        order.append(next_idx)
        remaining.remove(next_idx)
        current = next_idx

    return np.array(order)


def main():
    parser = argparse.ArgumentParser(
        description='Extract wall nodes and compute surface pressure coefficient.'
    )
    parser.add_argument('solution', type=Path, help='Path to the solution .dat file')
    parser.add_argument('--out', type=Path, help='Optional Tecplot output file')
    args = parser.parse_args()

    data = read_solution(args.solution)
    x, y, z = data[:, 0], data[:, 1], data[:, 2]
    rho, p = data[:, 3], data[:, 4]
    v1, v2, v3 = data[:, 5], data[:, 6], data[:, 7]

    rho_inf = rho[0]
    p_inf = p[0]
    u_inf = np.sqrt(v1[0]**2 + v2[0]**2 + v3[0]**2)
    cp = (p - p_inf) / (0.5 * rho_inf * u_inf**2)

    mask = z <= 0
    x_wall, y_wall, cp_wall = x[mask], y[mask], cp[mask]
    print(f'Total nodes: {len(x)}, wall nodes: {len(x_wall)}, excluded: {len(x) - len(x_wall)}')

    fig1, ax1 = plt.subplots()
    ax1.scatter(x_wall, y_wall, s=5)
    ax1.set_aspect('equal', adjustable='box')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Airfoil geometry (z<=0)')
    fig1.savefig('airfoil_geometry.png')

    order = nearest_neighbor_order(x_wall, y_wall)
    x_ord, y_ord, cp_ord = x_wall[order], y_wall[order], cp_wall[order]
    x_closed = np.append(x_ord, x_ord[0])
    y_closed = np.append(y_ord, y_ord[0])
    cp_closed = np.append(cp_ord, cp_ord[0])

    fig2, ax2 = plt.subplots()
    ax2.plot(x_closed, cp_closed)
    ax2.set_xlabel('x')
    ax2.set_ylabel('Cp')
    ax2.invert_yaxis()
    ax2.set_title('Surface Cp')
    fig2.savefig('surface_cp.png')

    if args.out:
        write_tecplot(args.out, x_closed, y_closed, cp_closed)

    # Show figures if interactive backend is available
    try:
        plt.show()
    except Exception:
        pass

if __name__ == '__main__':
    main()
