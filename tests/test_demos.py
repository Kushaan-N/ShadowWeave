"""Execute every module's ``__main__`` demo.

The project convention is that each module ships a runnable demo. Nothing enforced
that, so a demo could drift out of sync with the code it demonstrates and stay broken
indefinitely — which is exactly what happened to eval/baselines.py after its API
changed: the whole pytest suite stayed green while the demo raised AttributeError.

These run as subprocesses because the demos are __main__ blocks with side effects.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

# Every module that ships a demo. Anything with a `__main__` block belongs here.
DEMO_MODULES = [
    "shadowweave.console",
    "shadowweave.shadow.raycast",
    "shadowweave.shadow.bev",
    "shadowweave.shadow.occupancy",
    "shadowweave.shadow.zones",
    "shadowweave.shadow.visualizer",
    "shadowweave.world_model.unet",
    "shadowweave.world_model.ddpm",
    "shadowweave.world_model.dataset",
    "shadowweave.agents.local_agent",
    "shadowweave.agents.global_agent",
    "shadowweave.agents.orchestrator",
    "shadowweave.audio.cues",
    "shadowweave.audio.hrtf",
    "shadowweave.eval.metrics",
    "shadowweave.eval.baselines",
    "shadowweave.sim.usd_export",
    "shadowweave.ingestion.camera",
]


@pytest.mark.parametrize("module", DEMO_MODULES)
def test_demo_runs(module):
    r = subprocess.run(
        [sys.executable, "-m", module],
        capture_output=True, text=True, timeout=300,
    )
    assert r.returncode == 0, (
        f"{module} demo failed (exit {r.returncode})\n"
        f"--- stdout ---\n{r.stdout[-1500:]}\n--- stderr ---\n{r.stderr[-1500:]}"
    )


def test_every_main_block_is_covered():
    """A module with a __main__ block that is not listed above is untested."""
    import pathlib

    root = pathlib.Path(__file__).parents[1] / "shadowweave"
    have_main = set()
    for f in root.rglob("*.py"):
        if 'if __name__ == "__main__":' in f.read_text():
            rel = f.relative_to(root.parent).with_suffix("")
            have_main.add(str(rel).replace("/", "."))

    # The sim modules are covered by test_sim.py and are slow to run standalone.
    exempt = {"shadowweave.sim.mujoco_env", "shadowweave.sim.synthetic_data",
              "shadowweave.world_model.train", "shadowweave.eval.run_eval",
              "shadowweave.dashboard.app", "shadowweave.ingestion.depth"}
    missing = have_main - set(DEMO_MODULES) - exempt
    assert not missing, f"modules with demos that no test runs: {sorted(missing)}"
