"""FEA stress-map visualisation and animated GIF export.

Renders per-element Von Mises stress fields as two-panel matplotlib frames,
then stitches them into animated GIFs that show design convergence over the
course of an optimisation run.

Layout of each frame
────────────────────
 Left  │ Weapon geometry — solid blue fill, white holes, spin axis + CoM marked
 Right │ FEA stress field — triangles coloured by σ_VM / σ_yield
       │   blue  = low stress (over-designed / material can be removed)
       │   red   = near yield (critical; must keep material)

Requirements
────────────
  matplotlib  : pip install matplotlib          (for rendering frames)
  Pillow      : pip install Pillow              (for GIF assembly)

Both are optional; functions degrade gracefully if unavailable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from shapely.geometry import Polygon, MultiPolygon

from .config import WeaponConfig


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _draw_polygon(ax, poly: Polygon | MultiPolygon,
                  face: str = "#4a90e2", edge: str = "#2c5f8a") -> None:
    """Fill a shapely polygon on a matplotlib axis."""
    if isinstance(poly, MultiPolygon):
        for p in poly.geoms:
            _draw_polygon(ax, p, face, edge)
        return
    x, y = poly.exterior.xy
    ax.fill(x, y, fc=face, ec=edge, linewidth=0.8, alpha=0.85)
    for interior in poly.interiors:
        x, y = interior.xy
        ax.fill(x, y, fc="white", ec="#cc3333", linewidth=0.6, alpha=1.0)


def _draw_outline(ax, poly: Polygon | MultiPolygon) -> None:
    """Draw only the outline of a polygon (no fill)."""
    if isinstance(poly, MultiPolygon):
        for p in poly.geoms:
            _draw_outline(ax, p)
        return
    x, y = poly.exterior.xy
    ax.plot(x, y, color="black", linewidth=0.6, alpha=0.7)


# ---------------------------------------------------------------------------
# Single-frame renderer
# ---------------------------------------------------------------------------

def render_fea_frame(
    poly: Polygon | MultiPolygon,
    fea_result: dict,
    cfg: WeaponConfig,
    step_label: str,
    metrics: dict | None = None,
    save_path: Path | str | None = None,
    dpi: int = 90,
) -> Path | None:
    """Render a two-panel FEA frame and save it as a PNG.

    Parameters
    ----------
    poly        : assembled weapon polygon (with holes)
    fea_result  : dict from fea_stress_analysis_with_mesh() — must contain
                  'nodes', 'elements', 'vm_stresses', 'peak_stress_mpa',
                  'safety_factor'
    cfg         : weapon configuration (used for material / envelope info)
    step_label  : string appended to the title (e.g. "P1-042")
    metrics     : optional metrics dict for annotation
    save_path   : where to write the PNG (created if needed); None = dry-run
    dpi         : output resolution

    Returns the Path that was written, or None on failure.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")          # non-interactive backend for subprocess safety
        import matplotlib.pyplot as plt
        from matplotlib.collections import PolyCollection
        from matplotlib.colors import Normalize
    except ImportError:
        return None

    nodes       = fea_result.get("nodes")
    elements    = fea_result.get("elements")
    vm_stresses = fea_result.get("vm_stresses")

    if nodes is None or elements is None or vm_stresses is None or len(elements) == 0:
        return None

    yield_mpa    = cfg.material.yield_strength_mpa
    stress_ratio = (vm_stresses / max(yield_mpa, 1e-6)).clip(0.0, 1.2)

    # ── figure layout ────────────────────────────────────────────────────
    BG = "#12122a"
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 7))
    fig.patch.set_facecolor(BG)

    # ── LEFT: weapon geometry ────────────────────────────────────────────
    ax0.set_facecolor(BG)
    ax0.set_aspect("equal")
    _draw_polygon(ax0, poly)
    ax0.plot(0, 0, "wo", markersize=5, zorder=10, label="Spin axis")
    cx, cy = poly.centroid.x, poly.centroid.y
    ax0.plot(cx, cy, "r+", markersize=10, markeredgewidth=2, zorder=11, label="CoM")

    if metrics:
        mass_str   = f"Mass  {metrics.get('mass_kg', 0):.3f} kg"
        moi_str    = f"MOI   {metrics.get('moi_kg_mm2', 0):.0f} kg·mm²"
        energy_str = f"E     {metrics.get('energy_joules', 0):.0f} J"
        bite_str   = f"Bite  {metrics.get('bite_mm', 0):.1f} mm"
        n_str      = f"Teeth {int(metrics.get('n_teeth', metrics.get('num_teeth', 0)))}"
        ax0.set_title(
            f"Step {step_label}\n{mass_str}  {moi_str}\n{energy_str}  {bite_str}  {n_str}",
            color="white", fontsize=8.5, pad=5,
        )
    else:
        ax0.set_title(f"Step {step_label}", color="white", fontsize=9)

    ax0.tick_params(colors="white", labelsize=7)
    ax0.set_xlabel("mm", color="white", fontsize=8)
    ax0.set_ylabel("mm", color="white", fontsize=8)
    for sp in ax0.spines.values():
        sp.set_edgecolor("#333355")
    ax0.legend(facecolor="#1e1e3e", labelcolor="white", fontsize=7, loc="upper right")

    # ── RIGHT: FEA stress field ───────────────────────────────────────────
    ax1.set_facecolor(BG)
    ax1.set_aspect("equal")

    # Build a PolyCollection from the element triangles
    triangles = nodes[elements]          # (n_elem, 3, 2)
    norm = Normalize(vmin=0.0, vmax=1.0)
    cmap = plt.get_cmap("RdYlBu_r")

    col = PolyCollection(
        triangles,
        array=stress_ratio,
        cmap=cmap,
        norm=norm,
        edgecolors="none",
        antialiased=False,
    )
    ax1.add_collection(col)
    _draw_outline(ax1, poly)

    cbar = fig.colorbar(col, ax=ax1, fraction=0.03, pad=0.03)
    cbar.set_label("σ_VM / σ_yield", color="white", fontsize=8)
    cbar.ax.tick_params(colors="white", labelsize=7)
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")

    sf  = fea_result.get("safety_factor", float("inf"))
    pk  = fea_result.get("peak_stress_mpa", 0.0)
    sf_s = f"{sf:.2f}" if sf < 999 else ">999"
    ax1.set_title(
        f"Von Mises stress   σ_peak = {pk:.0f} MPa   SF = {sf_s}",
        color="white", fontsize=8.5, pad=5,
    )
    ax1.tick_params(colors="white", labelsize=7)
    ax1.set_xlabel("mm", color="white", fontsize=8)
    for sp in ax1.spines.values():
        sp.set_edgecolor("#333355")
    ax1.autoscale()

    plt.tight_layout(pad=1.2)

    if save_path is None:
        plt.close(fig)
        return None

    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# GIF assembler
# ---------------------------------------------------------------------------

def export_gif(
    frame_source: Path | Sequence[Path],
    output_path: Path | str,
    fps: int = 5,
    hold_last: int = 4,
) -> Path | None:
    """Stitch PNG frames into an animated GIF.

    Parameters
    ----------
    frame_source : a directory of PNG files, or an explicit list of Paths
    output_path  : destination .gif path
    fps          : frames per second (default 5 → 200 ms/frame)
    hold_last    : extra copies of the final frame (pause at end)

    Returns the output Path on success, None if Pillow is unavailable or no
    frames were found.
    """
    try:
        from PIL import Image
    except ImportError:
        print("[fea_viz] Pillow not installed — GIF export skipped.  "
              "Install with: pip install Pillow")
        return None

    if isinstance(frame_source, (str, Path)):
        frame_paths = sorted(Path(frame_source).glob("*.png"))
    else:
        frame_paths = sorted(frame_source)

    if not frame_paths:
        return None

    images = [Image.open(fp).convert("RGB") for fp in frame_paths]
    if not images:
        return None

    # Normalise all frames to the same pixel dimensions.
    # bbox_inches='tight' in matplotlib can produce slightly different sizes
    # per frame; mismatched sizes crash Pillow's GIF delta-encoder.
    target_w, target_h = images[0].size
    images = [img.resize((target_w, target_h), Image.LANCZOS) for img in images]
    images.extend([images[-1]] * hold_last)

    # Convert to palette-quantised RGBA for clean GIF output.
    # Using RGBA + transparency=0 avoids the RGB delta-paste bug in some
    # Pillow versions while keeping colour fidelity via quantize().
    palette_imgs = [img.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
                    for img in images]

    duration_ms = int(1000 / max(fps, 1))
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    palette_imgs[0].save(
        out,
        save_all=True,
        append_images=palette_imgs[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,   # avoid delta-frame bugs in older Pillow builds
    )
    return out
