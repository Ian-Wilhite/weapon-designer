"""Optional matplotlib preview of weapon profiles."""

from __future__ import annotations

from shapely.geometry import Polygon, MultiPolygon

from .config import WeaponConfig


def preview_weapon(
    weapon: Polygon | MultiPolygon,
    cfg: WeaponConfig,
    metrics: dict | None = None,
    score: float | None = None,
):
    """Show a matplotlib preview of the weapon profile."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import PathPatch
        from matplotlib.path import Path
        import numpy as np
    except ImportError:
        print("matplotlib not installed. Install with: pip install weapon-designer[viz]")
        return

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="k", linewidth=0.5)
    ax.axvline(x=0, color="k", linewidth=0.5)

    def plot_polygon(poly: Polygon):
        # Exterior
        x, y = poly.exterior.xy
        ax.fill(x, y, alpha=0.3, fc="steelblue", ec="navy", linewidth=1.5)
        # Holes
        for interior in poly.interiors:
            x, y = interior.xy
            ax.fill(x, y, alpha=1.0, fc="white", ec="red", linewidth=1.0)

    if isinstance(weapon, MultiPolygon):
        for geom in weapon.geoms:
            plot_polygon(geom)
    else:
        plot_polygon(weapon)

    # Mark centre of mass
    cx, cy = weapon.centroid.x, weapon.centroid.y
    ax.plot(cx, cy, "r+", markersize=12, markeredgewidth=2, label="CoM")
    ax.plot(0, 0, "ko", markersize=6, label="Spin axis")

    # Draw envelope
    if cfg.weapon_style == "bar":
        from matplotlib.patches import Rectangle
        half_l = cfg.envelope.max_length_mm / 2
        half_w = cfg.envelope.max_width_mm / 2
        rect = Rectangle(
            (-half_l, -half_w), cfg.envelope.max_length_mm, cfg.envelope.max_width_mm,
            fill=False, ec="gray", ls="--", linewidth=1.0, label="Envelope",
        )
        ax.add_patch(rect)
    else:
        circle = plt.Circle(
            (0, 0), cfg.envelope.max_radius_mm,
            fill=False, ec="gray", ls="--", linewidth=1.0, label="Envelope",
        )
        ax.add_patch(circle)

    # Title with stats
    title = f"{cfg.weapon_style.upper()} Weapon — {cfg.material.name}"
    if metrics:
        title += (
            f"\nMass: {metrics['mass_kg']:.3f} kg | "
            f"MOI: {metrics['moi_kg_mm2']:.1f} kg·mm² | "
            f"Energy: {metrics['energy_joules']:.0f} J"
        )
    if score is not None:
        title += f" | Score: {score:.4f}"
    ax.set_title(title)
    ax.set_xlabel("mm")
    ax.set_ylabel("mm")
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.show()
