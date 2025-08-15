"""Utilities for ordering nodes given various connectivity assumptions."""

from __future__ import annotations

import numpy as np


def order_points_from_lineseg(npts: int, elements: np.ndarray) -> list[int]:
    """Return node indices ordered along a polyline.

    Assumes ``elements`` describes a single chain of line segments where each
    node connects to at most two neighbors.  Segments may form an open or
    closed polyline; any unconnected nodes are appended at the end of the
    returned sequence.
    """
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
    """Return node indices outlining the boundary loop of a surface mesh.

    Assumes ``elements`` are polygonal elements of a manifold surface mesh.
    Edges used exactly once are treated as boundary edges and are walked to
    produce a loop around the boundary.  Degenerate cases or multiple boundary
    loops are not handled.
    """
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
    """Greedily order points by walking to the nearest unused neighbor.

    Assumes points roughly follow a curve so that a nearest-neighbor walk does
    not jump across disjoint regions.  No connectivity information is required
    and the path may self-intersect for scattered point sets.
    """
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
