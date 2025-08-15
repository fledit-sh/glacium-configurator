"""Merge wall zones in a mesh file.

This utility reads a mesh file with a ``zone_id`` cell-data array and
replaces the IDs of specified wall zones with a single merged ID.  The
resulting mesh is written to disk and an optional plot of zone counts is
saved using ``matplotlib``.

Example using command-line options::

    python -m src.merge_wall_zones --input examples/sample_mesh.vtp \
        --output examples/merged_mesh.vtp --wall-zones 1 2 \
        --plot examples/zone_counts.png

The same options can be supplied through a YAML configuration file::

    python -m src.merge_wall_zones --config examples/merge_config.yaml

The configuration keys mirror the command-line options: ``input``,
``output``, ``wall_zones`` and ``plot``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
import yaml


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Path to the mesh file.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("merged_wall.vtp"),
        help="Path for the merged mesh output.",
    )
    parser.add_argument(
        "--wall-zones",
        type=int,
        nargs="+",
        help="Zone IDs to merge into one wall zone.",
    )
    parser.add_argument(
        "--merged-id",
        type=int,
        default=None,
        help="ID assigned to the merged zone (defaults to the smallest ID).",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        help="Optional path to save a bar plot of zone counts.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="YAML file providing any of the command-line options.",
    )
    return parser.parse_args(argv)


def _load_config(path: Path | None) -> dict:
    if not path:
        return {}
    with open(path, "r", encoding="utf8") as f:
        return yaml.safe_load(f) or {}


def _merge_wall_zones(mesh: pv.DataSet, ids: Iterable[int], merged_id: int | None) -> None:
    zone_ids = np.array(mesh.cell_data.get("zone_id"))
    if zone_ids.size == 0:
        raise KeyError("Mesh is missing required 'zone_id' cell data")
    ids = list(ids)
    if merged_id is None:
        merged_id = int(min(ids))
    mask = np.isin(zone_ids, ids)
    zone_ids[mask] = merged_id
    mesh.cell_data["zone_id"] = zone_ids


def _plot_counts(before: np.ndarray, after: np.ndarray, path: Path) -> None:
    ids = np.union1d(before, after)
    counts_before = [np.count_nonzero(before == i) for i in ids]
    counts_after = [np.count_nonzero(after == i) for i in ids]

    fig, ax = plt.subplots()
    x = np.arange(len(ids))
    width = 0.4
    ax.bar(x - width / 2, counts_before, width, label="before")
    ax.bar(x + width / 2, counts_after, width, label="after")
    ax.set_xticks(x)
    ax.set_xticklabels(ids)
    ax.set_xlabel("zone_id")
    ax.set_ylabel("cell count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    cfg = _load_config(args.config)

    input_path = Path(args.input or cfg.get("input"))
    output_path = Path(cfg.get("output", args.output))
    wall_zones = args.wall_zones or cfg.get("wall_zones")
    merged_id = args.merged_id or cfg.get("merged_id")
    plot_path = args.plot or cfg.get("plot")

    if not input_path:
        raise ValueError("An input mesh path must be provided")
    if not wall_zones:
        raise ValueError("At least one wall zone ID must be provided")

    mesh = pv.read(input_path)
    before = np.array(mesh.cell_data.get("zone_id"))
    _merge_wall_zones(mesh, wall_zones, merged_id)
    after = np.array(mesh.cell_data.get("zone_id"))
    mesh.save(output_path)
    if plot_path:
        _plot_counts(before, after, Path(plot_path))


if __name__ == "__main__":
    main()
