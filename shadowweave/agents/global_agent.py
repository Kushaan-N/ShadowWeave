"""Global coordinator agent — A* planner on predicted occupancy graph at 2Hz."""

from __future__ import annotations

import heapq
from typing import Optional

import numpy as np
from omegaconf import DictConfig


class GlobalAgent:
    """Builds a dynamic graph over predicted occupancy and computes safe waypoints via A*.

    Primary method: ``plan(occupancy_pred, start, goal) -> waypoints``

    Args:
        occupancy_pred: (H, W) float32, predicted occupancy probability (0=free, 1=blocked)
        start: (row, col) integer grid coordinates
        goal:  (row, col) integer grid coordinates

    Returns:
        waypoints: list of (row, col) tuples from start to goal, or [] if no path found
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self._w = cfg.agents.global_agent if hasattr(cfg.agents, "global_agent") else None

    def plan(
        self,
        occupancy_pred: np.ndarray,
        start: tuple[int, int],
        goal: tuple[int, int],
    ) -> list[tuple[int, int]]:
        """A* on the predicted occupancy grid. Edge weight = obstacle probability."""
        H, W = occupancy_pred.shape
        if not (0 <= start[0] < H and 0 <= start[1] < W):
            return []
        if not (0 <= goal[0] < H and 0 <= goal[1] < W):
            return []

        def heuristic(a: tuple[int, int], b: tuple[int, int]) -> float:
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        open_heap: list[tuple[float, tuple[int, int]]] = []
        heapq.heappush(open_heap, (0.0, start))
        came_from: dict[tuple[int, int], Optional[tuple[int, int]]] = {start: None}
        g_score: dict[tuple[int, int], float] = {start: 0.0}

        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while open_heap:
            _, current = heapq.heappop(open_heap)
            if current == goal:
                return self._reconstruct(came_from, current)

            r, c = current
            for dr, dc in neighbors:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < H and 0 <= nc < W):
                    continue
                cost = float(occupancy_pred[nr, nc]) + 1e-3  # never zero
                tentative_g = g_score[current] + cost
                if (nr, nc) not in g_score or tentative_g < g_score[(nr, nc)]:
                    g_score[(nr, nc)] = tentative_g
                    f = tentative_g + heuristic((nr, nc), goal)
                    heapq.heappush(open_heap, (f, (nr, nc)))
                    came_from[(nr, nc)] = current

        return []  # no path found

    def _reconstruct(
        self,
        came_from: dict[tuple[int, int], Optional[tuple[int, int]]],
        current: tuple[int, int],
    ) -> list[tuple[int, int]]:
        path = []
        while current is not None:
            path.append(current)
            current = came_from[current]
        return list(reversed(path))


if __name__ == "__main__":
    from omegaconf import OmegaConf
    import pathlib

    cfg_path = pathlib.Path(__file__).parents[1] / "configs" / "default.yaml"
    cfg = OmegaConf.load(cfg_path)

    agent = GlobalAgent(cfg)

    H, W = 16, 16
    occ = np.random.rand(H, W).astype(np.float32) * 0.3  # mostly free
    # add a wall
    occ[6:10, 6:10] = 0.95

    start, goal = (0, 0), (15, 15)
    waypoints = agent.plan(occ, start, goal)
    print(f"GlobalAgent: {start} → {goal}")
    print(f"  Waypoints ({len(waypoints)}): {waypoints[:5]}{'...' if len(waypoints) > 5 else ''}")
    assert waypoints[0] == start and waypoints[-1] == goal
    print("GlobalAgent OK")
