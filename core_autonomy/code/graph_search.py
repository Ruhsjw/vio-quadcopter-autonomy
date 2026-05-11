from heapq import heappush, heappop
import numpy as np

from .occupancy_map import OccupancyMap


def graph_search(world, resolution, margin, start, goal, astar):
    """
    A* / Dijkstra search on voxel occupancy map.

    IMPORTANT: Default behavior is PURE A* (fast, stable).
    A lightweight (cached) proximity penalty is included but disabled by default.
    """

    occ_map = OccupancyMap(world, resolution, margin)

    start = np.asarray(start, dtype=float).reshape(3)
    goal = np.asarray(goal, dtype=float).reshape(3)
    resolution = np.asarray(resolution, dtype=float).reshape(3)

    start_idx = tuple(occ_map.metric_to_index(start))
    goal_idx = tuple(occ_map.metric_to_index(goal))

    # If start/goal in collision or out of bounds => no path
    if occ_map.is_occupied_index(start_idx) or occ_map.is_occupied_index(goal_idx):
        return None, 0

    goal_center = occ_map.index_to_metric_center(goal_idx)

    def h(idx):
        if not astar:
            return 0.0
        return float(np.linalg.norm(occ_map.index_to_metric_center(idx) - goal_center))

    # 26-connected neighbors
    neighbor_steps = []
    neighbor_costs = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                neighbor_steps.append((dx, dy, dz))
                step_vec = np.array([dx, dy, dz], dtype=float) * resolution
                neighbor_costs.append(float(np.linalg.norm(step_vec)))

    # Bias paths toward the middle of narrow passages.  This costs a little
    # more search time, but buys useful clearance once VIO/tracking error is
    # in the loop.
    use_penalty = False
    pen_r = 2
    lam = 1.2
    penalty_cache = {}

    def local_penalty(idx):
        if idx in penalty_cache:
            return penalty_cache[idx]
        x, y, z = idx
        occ = 0
        total = 0
        for dx in range(-pen_r, pen_r + 1):
            for dy in range(-pen_r, pen_r + 1):
                for dz in range(-pen_r, pen_r + 1):
                    j = (x + dx, y + dy, z + dz)
                    # out-of-bounds counts as occupied => pushes away from boundary
                    if occ_map.is_occupied_index(j):
                        occ += 1
                    total += 1
        val = occ / max(total, 1)
        penalty_cache[idx] = val
        return val

    g = {start_idx: 0.0}
    parent = {start_idx: None}

    pq = []
    tie = 0
    heappush(pq, (g[start_idx] + h(start_idx), tie, start_idx))

    closed = set()
    nodes_expanded = 0

    while pq:
        _, _, u = heappop(pq)

        if u in closed:
            continue
        closed.add(u)
        nodes_expanded += 1

        if u == goal_idx:
            # Reconstruct
            idx_path = []
            cur = u
            while cur is not None:
                idx_path.append(cur)
                cur = parent[cur]
            idx_path.reverse()

            pts = [start.copy()]
            for idx in idx_path[1:-1]:
                pts.append(occ_map.index_to_metric_center(idx))
            pts.append(goal.copy())
            return np.vstack(pts), nodes_expanded

        ux, uy, uz = u
        gu = g[u]

        for (dx, dy, dz), step_cost in zip(neighbor_steps, neighbor_costs):
            v = (ux + dx, uy + dy, uz + dz)

            if occ_map.is_occupied_index(v):
                continue

            edge_cost = step_cost
            if use_penalty:
                edge_cost = step_cost * (1.0 + lam * local_penalty(v))

            tentative = gu + edge_cost
            if tentative < g.get(v, float("inf")):
                g[v] = tentative
                parent[v] = u
                tie += 1
                heappush(pq, (tentative + h(v), tie, v))

    return None, nodes_expanded
