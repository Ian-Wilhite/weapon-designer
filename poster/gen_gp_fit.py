"""Rearrange gp_fit.png from 1×4 to 2×2 layout.

Source: poster/assets/Screenshot from 2026-03-18 16-17-45.png
(identical content to the original gp_fit.png at 1603×356 px).

Splits into 4 equal panels and recomposes as 2×2.
Saves to poster/assets/gp_fit.png
"""

from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

_HERE = Path(__file__).parent
SRC   = _HERE / "assets" / "Screenshot from 2026-03-18 16-17-45.png"
OUT   = _HERE / "assets" / "gp_fit.png"

img = np.array(Image.open(SRC).convert("RGB"))
h, w = img.shape[:2]   # 1603 × 356

# Detect header height by walking three phases:
#   1. skip initial white rows (top padding)
#   2. skip title text rows (first dark band)
#   3. skip white gap between title and axes
# Result: header_end is the first row of actual plot content.
row_means = img.mean(axis=(1, 2))
i = 0
while i < h and row_means[i] > 235:   # phase 1: top white padding
    i += 1
while i < h and row_means[i] <= 235:  # phase 2: title text
    i += 1
while i < h and row_means[i] > 235:   # phase 3: white gap after title
    i += 1
header_end = i

# Crop off the header so the suptitle is fully removed before splitting
img = img[header_end:, :, :]

# Equal-width panel splits (4 panels → boundaries at 0, 401, 803, 1203, 1603)
boundaries = [round(i * w / 4) for i in range(5)]
panels = [img[:, boundaries[i]:boundaries[i+1], :] for i in range(4)]

# ── 2×2 layout ────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(10, 8), dpi=150)
gs = GridSpec(2, 2, figure=fig,
              left=0.01, right=0.99, top=0.99, bottom=0.01,
              wspace=0.03, hspace=0.04)

positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
for (row, col), panel in zip(positions, panels):
    ax = fig.add_subplot(gs[row, col])
    ax.imshow(panel, aspect="auto", interpolation="bilinear")
    ax.axis("off")

fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: {OUT}")
