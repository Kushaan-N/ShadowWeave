"""Global coordinator agent — A* planner on the predicted BEV occupancy grid at 2Hz.

Three fixes over the first version:
  * It read ``cfg.agents.global_agent``, which does not exist (the key is
    ``agents.global``), so astar_heuristic_weight was silently never applied.
  * There was no closed set, so nodes were re-expanded repeatedly.
  * The heuristic returned Manhattan step count while edge costs ran from 1e-3 to
    1.0, making it wildly inadmissible — A* degenerated into a greedy walk and
    returned paths far from optimal. The heuristic is now scaled by the true minimum
    step cost, with the inflation factor applied explicitly and deliberately.
"""

from __future__ import annotations

import heapq
import math
from typing import Optional

import numpy as np
from omegaconf import DictConfig

_STEP_COST = 1.0  # baseline cost of moving one cell, before obstacle penalties


class GlobalAgent:
    """Builds a dynamic graph over predicted occupancy and computes safe waypoints via A*.

    Primary method: ``plan(occupancy_pred, start, goal, shadow) -> waypoints``

    Args:
        occupancy_pred: (H, W) float32, predicted occupancy probability (0=free, 1=blocked)
        start: (row, col) integer grid coordinates
        goal:  (row, col) integer grid coordinates
        shadow: optional (H, W) float32 in [0,1], 1 = unobserved. Routing through
                unobserved space is penalised rather than forbidden.

    Returns:
        waypoints: list of (row, col) tuples from start to goal, or [] if no path found
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        # OmegaConf stores this under the literal key "global", which is a Python
        # keyword and so cannot be reached with attribute access.
        g = cfg.agents["global"]
        self.heuristic_weight = float(g.astar_heuristic_weight)
        self.obstacle_cost = float(g.astar_obstacle_cost)
        self.shadow_cost = float(g.astar_shadow_cost)
        self.allow_diagonal = bool(g.astar_allow_diagonal)
        self.block_threshold = float(g.astar_block_threshold)

    def plan(
        self,
        occupancy_pred: np.ndarray,
        start: tuple[int, int],
        goal: tuple[int, int],
        shadow: Optional[np.ndarray] = None,
    ) -> list[tuple[int, int]]:
        H, W = occupancy_pred.shape
        for name, pt in (("start", start), ("goal", goal)):
            if int(pt[0]) != pt[0] or int(pt[1]) != pt[1]:
                raise ValueError(f"{name} must be integer grid coordinates, got {pt}")
        start = (int(start[0]), int(start[1]))
        goal = (int(goal[0]), int(goal[1]))
        if not (0 <= start[0] < H and 0 <= start[1] < W):
            return []
        if not (0 <= goal[0] < H and 0 <= goal[1] < W):
            return []

        # A NaN cell means the model broke, not that the cell is free — treat it as
        # blocked, or A* silently returns [] (NaN fails every < comparison).
        occ = np.nan_to_num(occupancy_pred, nan=1.0, posinf=1.0, neginf=0.0)
        blocked = occ >= self.block_threshold
        if blocked[goal] or blocked[start]:
            return []  # a goal inside an obstacle has no safe route by definition
        if start == goal:
            return [start]

        cost_map = _STEP_COST + self.obstacle_cost * np.clip(occ, 0.0, 1.0)
        if shadow is not None:
            cost_map = cost_map + self.shadow_cost * np.nan_to_num(
                np.clip(shadow, 0.0, 1.0), nan=1.0
            )

        if self.allow_diagonal:
            moves = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        else:
            moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        def heuristic(a: tuple[int, int]) -> float:
            dr, dc = abs(a[0] - goal[0]), abs(a[1] - goal[1])
            if self.allow_diagonal:
                # Octile distance is the exact cost of the cheapest unobstructed
                # 8-connected path, so scaling it by _STEP_COST stays admissible.
                base = (dr + dc) + (math.sqrt(2) - 2) * min(dr, dc)
            else:
                base = dr + dc
            return self.heuristic_weight * _STEP_COST * base

        open_heap: list[tuple[float, int, tuple[int, int]]] = [(heuristic(start), 0, start)]
        came_from: dict[tuple[int, int], Optional[tuple[int, int]]] = {start: None}
        g_score: dict[tuple[int, int], float] = {start: 0.0}
        closed: set[tuple[int, int]] = set()
        counter = 0

        while open_heap:
            _, _, current = heapq.heappop(open_heap)
            if current == goal:
                return self._reconstruct(came_from, current)
            if current in closed:
                continue
            closed.add(current)

            r, c = current
            for dr, dc in moves:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < H and 0 <= nc < W) or (nr, nc) in closed:
                    continue
                # Near-certain obstacles are impassable, not merely expensive — a
                # finite penalty meant a fully sealed wall still yielded a "path"
                # straight through it, reported to the caller as safe.
                if blocked[nr, nc]:
                    continue
                # Diagonal steps cover sqrt(2) cells of ground, so they must cost more
                # or the planner prefers staircases through obstacles.
                step = math.sqrt(2) if (dr and dc) else 1.0
                tentative = g_score[current] + float(cost_map[nr, nc]) * step
                if tentative < g_score.get((nr, nc), float("inf")):
                    g_score[(nr, nc)] = tentative
                    came_from[(nr, nc)] = current
                    counter += 1
                    heapq.heappush(open_heap, (tentative + heuristic((nr, nc)), counter, (nr, nc)))

        return []  # no path found

    def path_cost(
        self,
        occupancy_pred: np.ndarray,
        path: list[tuple[int, int]],
        shadow: Optional[np.ndarray] = None,
    ) -> float:
        """Cost of a path under the same model plan() optimises — including the
        sqrt(2) diagonal step weight and the shadow penalty, so ranking candidate
        paths by this value agrees with A*'s own ranking."""
        if not path:
            return float("inf")
        occ = np.nan_to_num(occupancy_pred, nan=1.0, posinf=1.0, neginf=0.0)
        total = 0.0
        for (pr, pc), (r, c) in zip(path[:-1], path[1:]):
            step = math.sqrt(2) if (pr != r and pc != c) else 1.0
            cell = _STEP_COST + self.obstacle_cost * float(np.clip(occ[r, c], 0.0, 1.0))
            if shadow is not None:
                cell += self.shadow_cost * float(np.clip(shadow[r, c], 0.0, 1.0))
            total += cell * step
        return float(total)

    def _reconstruct(
        self,
        came_from: dict[tuple[int, int], Optional[tuple[int, int]]],
        current: tuple[int, int],
    ) -> list[tuple[int, int]]:
        path = []
        node: Optional[tuple[int, int]] = current
        while node is not None:
            path.append(node)
            node = came_from[node]
        return list(reversed(path))


if __name__ == "__main__":
    from ..utils import load_config

    cfg = load_config()
    agent = GlobalAgent(cfg)
    print(f"GlobalAgent heuristic_weight={agent.heuristic_weight} "
          f"(read from agents['global'], previously unreachable)")

    H = W = 24
    rng = np.random.default_rng(0)
    occ = rng.random((H, W)).astype(np.float32) * 0.1
    # A wall spanning the room with a symmetric gap on each side, so the planner has
    # a genuine left-or-right choice to make.
    occ[10:14, :] = 0.98
    occ[10:14, 8:11] = 0.02   # left gap — nearer, so it wins on distance alone
    occ[10:14, 17:20] = 0.02  # right gap — the detour

    start, goal = (0, 12), (23, 12)
    path = agent.plan(occ, start, goal)
    print(f"  {start} → {goal}: {len(path)} waypoints, cost={agent.path_cost(occ, path):.1f}")
    assert path and path[0] == start and path[-1] == goal

    # The planner must route through a gap rather than straight through the wall.
    through_wall = sum(1 for r, c in path if occ[r, c] > 0.5)
    print(f"  cells crossed with occupancy>0.5: {through_wall}  (want 0)")
    assert through_wall == 0

    # Mark the left half unobserved; the planner should now prefer the right gap.
    shadow = np.zeros((H, W), dtype=np.float32)
    shadow[:, :12] = 1.0
    shadow_path = agent.plan(occ, start, goal, shadow=shadow)
    left_with = sum(1 for _, c in shadow_path if c < 12)
    left_without = sum(1 for _, c in path if c < 12)
    print(f"  cells in the shadowed half: {left_with} with penalty vs "
          f"{left_without} without  (want fewer)")
    assert left_with <= left_without

    print(f"  unreachable goal → {agent.plan(np.zeros((H, W), np.float32), (0, 0), (99, 99))}")
    print("GlobalAgent OK")
