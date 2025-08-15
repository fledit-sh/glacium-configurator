import re
from pathlib import Path
import argparse

import numpy as np
import matplotlib.pyplot as plt

def read_solution(path: Path):
    with open(path, 'r') as f:
        lines = f.readlines()
    zone_line = next(i for i, line in enumerate(lines) if line.lstrip().startswith('ZONE'))
    match = re.search(r'N=\s*(\d+)', lines[zone_line])
    data_lines = lines[zone_line + 1:]
    # Join lines without newline characters so numbers split across lines are
    # reconstructed correctly (some lines break within a number).
    text = ''.join(line.rstrip('\n') for line in data_lines)
    values = np.fromstring(text, sep=' ')
    # There are 21 variables per node in the file.
    n_vars = 21
    n_nodes = len(values) // n_vars
    data = values[: n_nodes * n_vars].reshape(n_nodes, n_vars)
    return data

def write_tecplot(path: Path, x: np.ndarray, y: np.ndarray, cp: np.ndarray):
    """Write arrays to Tecplot ASCII file."""
    with open(path, 'w') as f:
        f.write('TITLE = "Merged Wall Cp"\n')
        f.write('VARIABLES = "X" "Y" "Cp"\n')
        f.write(f'ZONE T="MergedWall", I={len(x)}, DATAPACKING=POINT\n')
        for xi, yi, ci in zip(x, y, cp):
            f.write(f"{xi} {yi} {ci}\n")


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

    order = np.argsort(x_wall)
    fig2, ax2 = plt.subplots()
    ax2.plot(x_wall[order], cp_wall[order])
    ax2.set_xlabel('x')
    ax2.set_ylabel('Cp')
    ax2.invert_yaxis()
    ax2.set_title('Surface Cp')
    fig2.savefig('surface_cp.png')

    if args.out:
        write_tecplot(args.out, x_wall[order], y_wall[order], cp_wall[order])

    # Show figures if interactive backend is available
    try:
        plt.show()
    except Exception:
        pass

if __name__ == '__main__':
    main()
