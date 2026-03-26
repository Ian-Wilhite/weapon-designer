"""Rearrange mode_shapes.png from 1×4 to 2×2 layout.

Source: outputs/rom_surrogate/mode_shapes.png
Saves to poster/assets/mode_shapes.png
"""

from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

_HERE = Path(__file__).parent
SRC   = _HERE.parent / "outputs" / "rom_surrogate" / "mode_shapes.png"
OUT   = _HERE / "assets" / "mode_shapes.png"

img = np.array(Image.open(SRC).convert("RGB"))
h, w = img.shape[:2]

boundaries = [round(i * w / 4) for i in range(5)]
panels = [img[:, boundaries[i]:boundaries[i+1], :] for i in range(4)]

fig = plt.figure(figsize=(10, 8), dpi=150)
gs = GridSpec(2, 2, figure=fig,
              left=0.01, right=0.99, top=0.99, bottom=0.01,
              wspace=0.03, hspace=0.04)

for (row, col), panel in zip([(0,0),(0,1),(1,0),(1,1)], panels):
    ax = fig.add_subplot(gs[row, col])
    ax.imshow(panel, aspect="auto", interpolation="bilinear")
    ax.axis("off")

fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: {OUT}")
