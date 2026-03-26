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
        del fig
        return None

    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    del fig
    return out


# ---------------------------------------------------------------------------
# Spiral contact visualiser
# ---------------------------------------------------------------------------

def render_spiral_contact_frame(
    poly: Polygon | MultiPolygon,
    contacts: list,
    r_start: float,
    cfg: "WeaponConfig",
    step_label: str = "",
    metrics: dict | None = None,
    save_path: Path | str | None = None,
    dpi: int = 90,
) -> Path | None:
    """Render the Archimedean spiral contact diagram for a weapon design.

    Two-panel figure (dark theme, matching ``render_fea_frame``):

    Left — Cartesian weapon cross-section with:
      • weapon outline filled in blue
      • r_start reference circle (dashed)
      • one spiral arc per contact (yellow → orange gradient from start to
        contact, dimmed arc beyond contact)
      • contact point markers (bright circles)
      • surface tangent line at each contact (cyan, ±15 mm)
      • force-direction arrows (red)

    Right — Polar bite-depth rose chart:
      • weapon radial profile as a closed curve
      • bar for each contact showing bite_depth at its angular position
      • annotated with contact_angle_cos (face-on quality)

    Parameters
    ----------
    poly      : assembled weapon polygon (holes already subtracted)
    contacts  : list of ContactResult from ``spiral_contact.analyse_contacts``
    r_start   : starting radius used for all spirals (mm)
    cfg       : weapon configuration (rpm, approach speed)
    step_label: string appended to figure title
    metrics   : optional metrics dict (for annotation overlay)
    save_path : PNG output path; None = dry-run / return None
    dpi       : output resolution
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyArrowPatch
        import matplotlib.gridspec as gridspec
    except ImportError:
        return None

    if isinstance(poly, MultiPolygon):
        _outer = max(poly.geoms, key=lambda p: p.area)
    else:
        _outer = poly

    BG = "#12122a"
    fig = plt.figure(figsize=(14, 7), facecolor=BG)
    gs  = gridspec.GridSpec(1, 2, figure=fig)
    ax0 = fig.add_subplot(gs[0])                         # Cartesian panel
    ax1 = fig.add_subplot(gs[1], projection="polar")     # Polar rose panel

    # ── LEFT: weapon + spirals ────────────────────────────────────────────
    ax0.set_facecolor(BG)
    ax0.set_aspect("equal")
    _draw_polygon(ax0, poly)

    # r_start reference circle
    _theta_c = np.linspace(0, 2 * np.pi, 300)
    ax0.plot(r_start * np.cos(_theta_c), r_start * np.sin(_theta_c),
             color="#555577", ls="--", lw=0.7, alpha=0.6, zorder=2)

    # Spiral parameters
    omega      = 2.0 * np.pi * max(cfg.rpm, 1.0) / 60.0
    v_mps      = float(getattr(cfg.optimization, "drive_speed_mps", 6.0))
    v_per_rad  = v_mps * 1000.0 / omega        # mm/radian
    n_arc      = 600                            # points along spiral arc

    # Colour palette: one shade per contact (yellow → red gradient)
    n_c = max(len(contacts), 1)
    cmap_spirals = plt.get_cmap("YlOrRd")

    for ci, c in enumerate(contacts):
        col = cmap_spirals(0.3 + 0.65 * ci / n_c)

        # Arc from θ₀ to θ_contact (approaching phase)
        arc_len  = float(c.theta_contact - c.theta_0)
        if arc_len < 1e-6:
            arc_len = 0.01
        t_arc    = np.linspace(0.0, arc_len, n_arc)
        theta_arc = c.theta_0 + t_arc
        r_arc    = r_start - v_per_rad * t_arc
        r_arc    = np.clip(r_arc, 0.0, None)
        ax0.plot(r_arc * np.cos(theta_arc), r_arc * np.sin(theta_arc),
                 color=col, lw=0.9, alpha=0.75, zorder=4)

        # Faded tail past contact (shows where spiral would continue)
        tail_extra = min(0.5, arc_len * 0.25)
        t_tail     = np.linspace(arc_len, arc_len + tail_extra, 80)
        theta_tail = c.theta_0 + t_tail
        r_tail     = r_start - v_per_rad * t_tail
        r_tail     = np.clip(r_tail, 0.0, None)
        ax0.plot(r_tail * np.cos(theta_tail), r_tail * np.sin(theta_tail),
                 color=col, lw=0.5, alpha=0.22, zorder=3)

        # Contact point marker
        xc, yc = c.xy_contact
        ax0.scatter([xc], [yc], s=55, color=col, edgecolors="white",
                    linewidths=0.8, zorder=9)

        # Surface tangent line  (cyan, ±15 mm from contact point)
        nx, ny = c.outward_normal
        tx, ty = -ny, nx            # 90° rotation of normal = tangent
        L = 15.0
        ax0.plot([xc - L * tx, xc + L * tx],
                 [yc - L * ty, yc + L * ty],
                 color="#00CCCC", lw=1.6, alpha=0.85, zorder=10,
                 solid_capstyle="round")

        # Force direction arrow (red, 14 mm display length)
        fd_x, fd_y = c.force_direction
        arrow_scale = 14.0
        ax0.annotate(
            "",
            xy=(xc + arrow_scale * fd_x, yc + arrow_scale * fd_y),
            xytext=(xc, yc),
            arrowprops=dict(
                arrowstyle="->",
                color="#FF4444",
                lw=1.4,
                mutation_scale=10,
            ),
            zorder=11,
        )

    # Spin axis
    ax0.plot(0, 0, "wo", markersize=5, zorder=12)

    # Title / annotations
    if metrics:
        bite_str    = f"bite={metrics.get('bite_mm', 0):.1f} mm"
        n_str       = f"n_contacts={metrics.get('n_contacts', 0)}"
        cq_str      = f"quality={metrics.get('contact_quality', 0):.2f}"
        title_extra = f"{bite_str}  {n_str}  {cq_str}"
    else:
        n_contact_shown = len(contacts)
        mean_bite = float(np.mean([c.bite_depth for c in contacts])) if contacts else 0.0
        mean_cq   = float(np.mean([c.contact_angle_cos for c in contacts])) if contacts else 0.0
        title_extra = (f"n_contacts={n_contact_shown}  "
                       f"mean_bite={mean_bite:.1f} mm  quality={mean_cq:.2f}")

    ax0.set_title(
        f"Spiral Contact Diagram   {step_label}\n{title_extra}\n"
        "— cyan: surface tangent  → red: force direction",
        color="white", fontsize=8.5, pad=5,
    )
    ax0.tick_params(colors="white", labelsize=7)
    ax0.set_xlabel("mm", color="white", fontsize=8)
    ax0.set_ylabel("mm", color="white", fontsize=8)
    for sp in ax0.spines.values():
        sp.set_edgecolor("#333355")
    ax0.autoscale()

    # ── RIGHT: polar rose — bite depth at each contact angle ──────────────
    ax1.set_facecolor(BG)
    ax1.tick_params(colors="#aaaaaa", labelsize=6)
    ax1.spines["polar"].set_edgecolor("#333355")

    # Weapon radial profile (outer boundary in polar coords from centroid)
    _ext  = np.array(_outer.exterior.coords[:-1])
    _cx, _cy = _outer.centroid.x, _outer.centroid.y
    _angles_prof = np.arctan2(_ext[:, 1] - _cy, _ext[:, 0] - _cx)
    _radii_prof  = np.hypot(_ext[:, 0] - _cx, _ext[:, 1] - _cy)
    _sort        = np.argsort(_angles_prof)
    _ap, _rp     = _angles_prof[_sort], _radii_prof[_sort]
    # Close the loop
    _ap = np.append(_ap, _ap[0] + 2 * np.pi)
    _rp = np.append(_rp, _rp[0])
    ax1.plot(_ap, _rp, color="#4a90e2", lw=1.2, alpha=0.7, zorder=3)
    ax1.fill(_ap, _rp, color="#4a90e2", alpha=0.15, zorder=2)

    if contacts:
        _c_angles = np.array([c.theta_contact % (2 * np.pi) for c in contacts])
        _c_depths = np.array([c.bite_depth for c in contacts])
        _c_quals  = np.array([c.contact_angle_cos for c in contacts])

        # Bar width: 2π / max(n_contacts, 8) so bars don't overlap
        bar_w = min(2 * np.pi / max(len(contacts), 8), 0.35)

        # ── Scale bars so the tallest fills ~30% of the outer chart ring ────
        # Bars start at r_start (weapon envelope boundary) and extend outward.
        # This decouples the bite_depth scale from the weapon-profile scale so
        # that even sub-millimetre bites are clearly visible (the weapon profile
        # can be 100–180 mm but bite depths are typically 0.5–5 mm).
        _max_depth = float(max(_c_depths)) if len(_c_depths) > 0 else 1.0
        _max_depth = max(_max_depth, 1e-6)
        # Outer limit: 1.35 × r_start; bars fill 30% of that outer region
        _r_outer  = r_start * 1.35
        _bar_scale = (_r_outer - r_start) * 0.30 / _max_depth
        _scaled_depths = _c_depths * _bar_scale

        _bars = ax1.bar(
            _c_angles, _scaled_depths, width=bar_w,
            bottom=r_start, alpha=0.80,
            color=[plt.get_cmap("YlOrRd")(0.3 + 0.65 * q) for q in _c_quals],
            edgecolor="#222244", linewidth=0.4,
            zorder=5,
        )

        # Annotate each bar: show actual bite_depth (mm) and contact quality
        for ang, dep, sdep, q in zip(_c_angles, _c_depths, _scaled_depths, _c_quals):
            ax1.text(ang, r_start + sdep + r_start * 0.02,
                     f"{dep:.2f}mm\ncq={q:.2f}",
                     ha="center", va="bottom",
                     fontsize=5.5, color="white", zorder=6)

        ax1.set_ylim(0, _r_outer * 1.18)

    ax1.set_title(
        "Bite depth per contact angle\n(bars from weapon boundary, actual mm labelled,\n"
        "colour = contact quality  0=glancing → 1=face-on)",
        color="white", fontsize=7.5, pad=10,
    )
    ax1.yaxis.label.set_color("white")

    plt.tight_layout(pad=1.2)

    if save_path is None:
        plt.close(fig)
        del fig
        return None

    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    del fig
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
