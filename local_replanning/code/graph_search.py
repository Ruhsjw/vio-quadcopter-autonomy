from heapq import heappush, heappop
import numpy as np


def graph_search(occ_map, start, goal, astar):
    """
    Search the provided occupancy map from start to goal.

    EC passes in a local OccupancyMap object. The cropped local goal can fall
    inside an obstacle, so when the exact goal cell is occupied we accept the
    closest free cell within a small metric tolerance.
    """
    start = np.asarray(start, dtype=float).reshape(3)
    goal = np.asarray(goal, dtype=float).reshape(3)
    resolution = np.asarray(occ_map.resolution, dtype=float).reshape(3)

    start_index = tuple(occ_map.metric_to_index(start))
    goal_index = tuple(occ_map.metric_to_index(goal))

    start_index = _nearest_free_index(occ_map, start_index, max_radius=4)
    if start_index is None:
        return None, 0

    goal_occupied = occ_map.is_occupied_index(goal_index)
    goal_center = goal.copy()
    goal_tol = max(0.75, 3.0 * float(np.max(resolution)))

    neighbor_steps = []
    neighbor_costs = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                step = (dx, dy, dz)
                neighbor_steps.append(step)
                neighbor_costs.append(float(np.linalg.norm(resolution * np.array(step))))

    def heuristic(idx):
        if not astar:
            return 0.0
        p = occ_map.index_to_metric_center(idx)
        return float(np.linalg.norm(p - goal_center))

    def reached(idx):
        if not goal_occupied and idx == goal_index:
            return True
        if goal_occupied:
            p = occ_map.index_to_metric_center(idx)
            return float(np.linalg.norm(p - goal_center)) <= goal_tol
        return False

    g = {start_index: 0.0}
    parent = {start_index: None}
    pq = []
    tie = 0
    heappush(pq, (heuristic(start_index), tie, start_index))

    closed = set()
    nodes_expanded = 0
    best_idx = start_index
    best_h = heuristic(start_index)

    max_expansions = 8000

    while pq and nodes_expanded < max_expansions:
        _, _, u = heappop(pq)
        if u in closed:
            continue
        closed.add(u)
        nodes_expanded += 1

        hu = heuristic(u)
        if hu < best_h:
            best_h = hu
            best_idx = u

        if reached(u):
            return _reconstruct_path(occ_map, parent, u, start, goal,
                                     exact_goal=(u == goal_index and not goal_occupied)), nodes_expanded

        ux, uy, uz = u
        gu = g[u]
        for (dx, dy, dz), step_cost in zip(neighbor_steps, neighbor_costs):
            v = (ux + dx, uy + dy, uz + dz)
            if v in closed or occ_map.is_occupied_index(v):
                continue
            tentative = gu + step_cost
            if tentative < g.get(v, float("inf")):
                g[v] = tentative
                parent[v] = u
                tie += 1
                heappush(pq, (tentative + heuristic(v), tie, v))

    if best_idx in parent and best_idx != start_index:
        return _reconstruct_path(occ_map, parent, best_idx, start, goal,
                                 exact_goal=False), nodes_expanded
    return None, nodes_expanded


def _nearest_free_index(occ_map, index, max_radius=4):
    if not occ_map.is_occupied_index(index):
        return index
    ix, iy, iz = index
    best = None
    best_d2 = float("inf")
    for r in range(1, int(max_radius) + 1):
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                for dz in range(-r, r + 1):
                    if max(abs(dx), abs(dy), abs(dz)) != r:
                        continue
                    cand = (ix + dx, iy + dy, iz + dz)
                    if occ_map.is_occupied_index(cand):
                        continue
                    d2 = dx * dx + dy * dy + dz * dz
                    if d2 < best_d2:
                        best = cand
                        best_d2 = d2
        if best is not None:
            return best
    return None


def _reconstruct_path(occ_map, parent, end_idx, start, goal, exact_goal):
    idx_path = []
    cur = end_idx
    while cur is not None:
        idx_path.append(cur)
        cur = parent[cur]
    idx_path.reverse()

    pts = [np.asarray(start, dtype=float).copy()]
    for idx in idx_path[1:-1]:
        pts.append(occ_map.index_to_metric_center(idx))
    if exact_goal:
        pts.append(np.asarray(goal, dtype=float).copy())
    else:
        pts.append(occ_map.index_to_metric_center(end_idx))
    return np.vstack(pts)
