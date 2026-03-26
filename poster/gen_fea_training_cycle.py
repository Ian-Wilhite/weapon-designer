"""Compose fea_training_cycle.png in a 3+2 layout from individual FEA frames.

Source: outputs/profile_sweep_8h/heavyweight_disk_bspline/frames_p1/
Uses steps 1, 20, 40, 60 and the last available frame.

Saves to poster/assets/fea_training_cycle.png
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

_HERE      = Path(__file__).parent
_FRAMES    = _HERE.parent / "outputs" / "profile_sweep_8h" / "heavyweight_disk_bspline" / "frames_p1"
OUT        = _HERE / "assets" / "fea_training_cycle.png"

# ── Select frames ─────────────────────────────────────────────────────────────
# Frame index = step - 1.  Run only reached step 67, so use 1,20,40,60,67.
FRAME_INDICES = [0, 19, 39, 59, 66]   # → steps 1, 20, 40, 60, 67

panels = []
labels = []
for fi in FRAME_INDICES:
    img_path  = _FRAMES / f"frame_{fi:04d}.png"
    meta_path = _FRAMES / f"frame_{fi:04d}_meta.json"
    panels.append(np.array(Image.open(img_path).convert("RGB")))
    with open(meta_path) as f:
        meta = json.load(f)
    step  = meta.get("step", fi + 1)
    score = meta.get("score", float("nan"))
    labels.append(f"Step {step}  |  Score: {score:.3f}")

# ── Compose 3+2 layout ────────────────────────────────────────────────────────
# 6-column GridSpec: top row uses all 6 cols (3 panels × 2), bottom row
# centres 2 panels at cols 1-2 and 3-4.
fig = plt.figure(figsize=(18, 8), dpi=150)
gs = GridSpec(2, 6, figure=fig,
              left=0.01, right=0.99, top=0.92, bottom=0.01,
              wspace=0.04, hspace=0.08)

specs = [
    gs[0, 0:2], gs[0, 2:4], gs[0, 4:6],   # top row
    gs[1, 1:3], gs[1, 3:5],                 # bottom row centred
]

for ax_spec, panel, label in zip(specs, panels, labels):
    ax = fig.add_subplot(ax_spec)
    ax.imshow(panel, aspect="auto", interpolation="bilinear")
    ax.set_title(label, fontsize=11, fontweight="bold", pad=4)
    ax.axis("off")

fig.suptitle(
    "FEA Stress Evolution — Heavyweight Disk Phase 1 (B-spline, 8 000 RPM)",
    fontsize=13, fontweight="bold", y=0.98,
)

fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: {OUT}")
