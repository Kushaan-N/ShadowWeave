#!/bin/bash
# Gate — verify the rollouts are not degenerate. Run after Phase 1 finishes.
# Pure numpy, so it is fine on the login node. Untracked helper; safe to delete.
set -euo pipefail
cd /work/pi_sniekum_umass_edu/kn/ShadowWeave

export WS=/scratch4/workspace/knaskar_umass_edu-shadowweave
export SW_VENV=$WS/sw-venv
export SW_DATA_ROOT=$WS/rollouts

source "$SW_VENV/bin/activate"
python - <<'EOF'
import numpy as np, glob, os
root = os.environ["SW_DATA_ROOT"]
for split in ("train", "val"):
    n = len(sorted(glob.glob(f"{root}/{split}/*.npz")))
    print(f"{split}: {n} episodes")
    assert n >= 1, f"no {split} episodes at {root}/{split}"
d = np.load(sorted(glob.glob(f"{root}/train/*.npz"))[0])
assert "bev_occupancy" in d.files, f"WRONG SCHEMA: {sorted(d.files)}"
t_std = d["target"].std(axis=0).mean()
h_std = d["target"].std(axis=1).mean()
flow  = (d["bev_flow"] != 0).mean()
print(f"target std over time  : {t_std:.5f}   (must be > 1e-3)")
print(f"target std over horizon: {h_std:.5f}  (>0 for moving/debris tiers)")
print(f"flow nonzero frac     : {flow:.4f}   (must be > 0)")
assert t_std > 1e-3 and flow > 0, "DEGENERATE — do NOT train; investigate the sim/EGL"
print("\n*** DATA OK — safe to train ***")
EOF
du -sh "$SW_DATA_ROOT"
