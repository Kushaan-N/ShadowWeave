.PHONY: help setup test smoke bench data train rl eval dashboard preflight demos clean clean-data

PYTHON ?= python
EPISODES ?= 400
VAL_EPISODES ?= 80

help:
	@echo "ShadowWeave"
	@echo "  make setup       install the package in editable mode"
	@echo "  make test        run the pytest suite"
	@echo "  make smoke       end-to-end pipeline smoke test (no training needed)"
	@echo "  make bench       throughput/memory benchmark to size batch and workers"
	@echo "  make demos       run every module's standalone demo"
	@echo "  make preflight   verify this machine/node can run everything"
	@echo "  make data        generate rollouts (EPISODES=$(EPISODES))"
	@echo "  make train       train the world model"
	@echo "  make rl          train the local agent with PPO"
	@echo "  make eval        run the evaluation harness"
	@echo "  make dashboard   launch the live Gradio dashboard"
	@echo "  make clean       remove caches and build artefacts"
	@echo "  make clean-data  remove generated rollouts and memmap caches"

setup:
	$(PYTHON) -m pip install -e ".[sim,viz,audio,depth,dev]"

test:
	$(PYTHON) -m pytest -q

smoke:
	$(PYTHON) scripts/pipeline_test.py --frames 10

bench:
	$(PYTHON) scripts/benchmark.py --data $(shell $(PYTHON) -c "print('data/rollouts')")

preflight:
	$(PYTHON) slurm/preflight.py

demos:
	@for m in shadowweave.shadow.raycast shadowweave.shadow.bev shadowweave.shadow.occupancy \
	          shadowweave.shadow.zones shadowweave.shadow.visualizer \
	          shadowweave.world_model.diffusion shadowweave.world_model.dataset \
	          shadowweave.agents.local_agent shadowweave.agents.global_agent \
	          shadowweave.agents.orchestrator shadowweave.audio.cues shadowweave.audio.hrtf \
	          shadowweave.eval.metrics shadowweave.sim.mujoco_env; do \
		echo "── $$m ──"; $(PYTHON) -m $$m >/dev/null 2>&1 && echo "  OK" || echo "  FAILED"; \
	done

data:
	$(PYTHON) -m shadowweave.sim.synthetic_data --episodes $(EPISODES) --split train
	$(PYTHON) -m shadowweave.sim.synthetic_data --episodes $(VAL_EPISODES) --split val --seed0 900000

train:
	$(PYTHON) -m shadowweave.world_model.train

rl:
	$(PYTHON) train_rl.py

eval:
	$(PYTHON) -m shadowweave.eval.run_eval

dashboard:
	$(PYTHON) -m shadowweave.dashboard.app --live

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type f -name '*.py[co]' -delete
	rm -rf .pytest_cache build dist *.egg-info MUJOCO_LOG.TXT

clean-data:
	rm -rf data/rollouts/*/.memmap
	@echo "Removed memmap caches. Delete data/rollouts/* by hand to drop the rollouts."
