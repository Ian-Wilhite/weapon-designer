"""Archimedean spiral contact analysis — comparison figure.

Shows ~20 spiral trajectories on a smooth disk and a 3-tooth weapon,
marks first-contact points, and draws impact force arrows.  A third
panel compares the bite-depth (r_start − r_contact) distributions.

Force arrows show the spiral path tangent dP/dθ at each contact point —
the velocity of the opponent in the weapon frame, and therefore the direction
of the impact force on the weapon.

Spiral equation:  r_enemy(θ) = r_start − v_per_rad · (θ − θ₀)
    v_per_rad = v_approach / ω   [mm / radian of weapon rotation]

Usage:
    python3 docs/figure_spiral_contacts.py

Output:
    docs/spiral_contacts.png
"""

from __future__ import annotations

import os
import sys

# Allow running from project root without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import FancyArrowPatch
from shapely.geometry import Polygon

from weapon_designer.spiral_contact import (
    profile_polar,
    find_first_contact,
    analyse_contacts,
)


# ── Parameters ────────────────────────────────────────────────────────────────

N_SPIRALS = 20
V_MS      = 10.0   # opponent approach speed [m/s]
RPM       = 1000.0  # weapon rotational speed [rpm]

# Derived
OMEGA     = RPM * 2.0 * np.pi / 60.0     # rad/s
V_PER_RAD = (V_MS * 1000.0) / OMEGA       # mm / radian of weapon rotation


# ── Weapon shapes ──────────────────────────────────────────────────────────────

def make_smooth_disk(R: float = 80.0, n_pts: int = 360) -> Polygon:
    """Circular disk profile."""
    theta = np.linspace(0.0, 2.0 * np.pi, n_pts, endpoint=False)
    coords = list(zip((R * np.cos(theta)).tolist(), (R * np.sin(theta)).tolist()))
    coords.append(coords[0])
    return Polygon(coords)


def make_toothed_weapon(
    R_mean: float = 70.0,
    amplitude: float = 22.0,
    n_teeth: int = 3,
    n_pts: int = 720,
) -> Polygon:
    """Star-shaped weapon: r(θ) = R_mean + amplitude·cos(n_teeth·θ)."""
    theta = np.linspace(0.0, 2.0 * np.pi, n_pts, endpoint=False)
    r = R_mean + amplitude * np.cos(n_teeth * theta)
    coords = list(zip((r * np.cos(theta)).tolist(), (r * np.sin(theta)).tolist()))
    coords.append(coords[0])
    p = Polygon(coords)
    return p.buffer(0) if not p.is_valid else p


# ── Spiral path (Cartesian) ───────────────────────────────────────────────────

def spiral_path(
    theta_0: float,
    r_start: float,
    theta_end: float,
    n_pts: int = 300,
) -> tuple[np.ndarray, np.ndarray]:
    """Cartesian coordinates of the spiral from θ₀ to θ_end."""
    theta = np.linspace(theta_0, theta_end, n_pts)
    r = np.maximum(r_start - V_PER_RAD * (theta - theta_0), 0.0)
    return r * np.cos(theta), r * np.sin(theta)


# ── Panel drawing ─────────────────────────────────────────────────────────────

def draw_panel(ax, poly: Polygon, contacts, r_start: float, title: str) -> None:
    """Render one weapon panel: profile + spirals + contacts + force arrows."""
    # --- weapon fill ---
    ext = np.array(poly.exterior.coords)
    ax.fill(ext[:, 0], ext[:, 1], fc="#4a90e2", ec="#2c5f8a",
            lw=1.0, alpha=0.75, zorder=1)

    # --- bore placeholder ---
    bore_r = 12.7  # 25.4 mm bore / 2
    theta_circ = np.linspace(0, 2 * np.pi, 120)
    ax.fill(bore_r * np.cos(theta_circ), bore_r * np.sin(theta_circ),
            fc="white", ec="#888888", lw=0.6, zorder=2)

    # --- spirals ---
    cmap = plt.cm.hsv
    norm = Normalize(vmin=0.0, vmax=2.0 * np.pi)

    bite_depths = np.array([c.bite_depth for c in contacts])
    max_bite = bite_depths.max() if len(bite_depths) else 1.0

    for c in contacts:
        color = cmap(norm(c.theta_0 % (2.0 * np.pi)))
        sx, sy = spiral_path(c.theta_0, r_start, c.theta_contact)
        ax.plot(sx, sy, color=color, lw=0.9, alpha=0.55, zorder=3)
        # Starting dot on the outer ring
        ax.plot(sx[0], sy[0], "o", color=color, ms=3.0, alpha=0.7, zorder=4)
        # Contact marker (size encodes bite depth)
        ms = 5 + 8 * (c.bite_depth / max(max_bite, 1.0))
        ax.plot(*c.xy_contact, "o", color=color, ms=ms,
                mec="white", mew=0.8, zorder=6)

    # --- force arrows (tangent to spiral path at contact = dP/dθ direction) ---
    arrow_len = r_start * 0.17
    for c in contacts:
        cx, cy = c.xy_contact
        dx = c.force_direction[0] * arrow_len
        dy = c.force_direction[1] * arrow_len
        ax.annotate(
            "",
            xy=(cx + dx, cy + dy),
            xytext=(cx, cy),
            arrowprops=dict(arrowstyle="->", color="#cc3333", lw=1.2),
            zorder=7,
        )

    # --- reference circles ---
    R_ring = np.array([c.r_contact for c in contacts])
    r_outer = float(r_start)
    for r_ref, style, lbl in [
        (r_outer, ":", "r_start"),
    ]:
        t = np.linspace(0, 2 * np.pi, 360)
        ax.plot(r_ref * np.cos(t), r_ref * np.sin(t),
                linestyle=style, color="gray", lw=0.7, alpha=0.6,
                label=f"{lbl} = {r_ref:.0f} mm", zorder=1)

    r_min_c = float(R_ring.min()) if len(R_ring) else r_outer
    r_max_c = float(R_ring.max()) if len(R_ring) else r_outer
    t = np.linspace(0, 2 * np.pi, 360)
    ax.plot(r_min_c * np.cos(t), r_min_c * np.sin(t),
            "--", color="#22aa22", lw=0.8, alpha=0.7,
            label=f"r_min contact = {r_min_c:.0f} mm", zorder=1)
    if r_max_c - r_min_c > 2:
        ax.plot(r_max_c * np.cos(t), r_max_c * np.sin(t),
                "--", color="orange", lw=0.8, alpha=0.7,
                label=f"r_max contact = {r_max_c:.0f} mm", zorder=1)

    # --- spin direction indicator ---
    ang = np.pi / 4
    r_ann = r_outer * 0.45
    ax.annotate(
        "",
        xy=(r_ann * np.cos(ang + 0.35), r_ann * np.sin(ang + 0.35)),
        xytext=(r_ann * np.cos(ang), r_ann * np.sin(ang)),
        arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
        zorder=8,
    )
    ax.text(
        r_ann * np.cos(ang + 0.17),
        r_ann * np.sin(ang + 0.17) - r_outer * 0.10,
        "ω", fontsize=12, ha="center", color="black",
    )

    lim = r_outer * 1.18
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("x (mm)", fontsize=9)
    ax.set_ylabel("y (mm)", fontsize=9)
    ax.axhline(0, color="gray", lw=0.3, alpha=0.3)
    ax.axvline(0, color="gray", lw=0.3, alpha=0.3)
    ax.legend(fontsize=7, loc="lower right", framealpha=0.6)


def draw_distribution(ax, contacts_s, contacts_t, r_start_s, r_start_t) -> None:
    """Compare bite-depth distributions between smooth and toothed weapons."""
    bite_s = [c.bite_depth for c in contacts_s]
    bite_t = [c.bite_depth for c in contacts_t]

    # Box plot
    bp = ax.boxplot(
        [bite_s, bite_t],
        tick_labels=["Smooth\nDisk", "3-Tooth\nWeapon"],
        patch_artist=True,
        widths=0.4,
        boxprops=dict(facecolor="#4a90e2", alpha=0.6),
        medianprops=dict(color="black", lw=2.0),
        whiskerprops=dict(color="black"),
        capprops=dict(color="black"),
        flierprops=dict(marker="x", color="#888888", ms=4),
    )
    bp["boxes"][1].set_facecolor("#cc6644")

    # Individual points (jittered)
    rng = np.random.default_rng(42)
    j = 0.06
    ax.scatter(
        1 + rng.uniform(-j, j, len(bite_s)),
        bite_s, alpha=0.7, s=25, color="#2c5f8a", zorder=5,
    )
    ax.scatter(
        2 + rng.uniform(-j, j, len(bite_t)),
        bite_t, alpha=0.7, s=25, color="#aa3322", zorder=5,
    )

    ax.set_ylabel("Bite depth  (r_start − r_contact)  [mm]", fontsize=9)
    ax.set_title("Bite-Depth Distribution", fontsize=12, fontweight="bold")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    # Annotations
    for i, (bites, color) in enumerate([(bite_s, "#2c5f8a"), (bite_t, "#aa3322")], 1):
        mb = max(bites) if bites else 0.0
        ax.text(
            i, mb + ax.get_ylim()[1] * 0.02 if mb > 0 else 1,
            f"max = {mb:.1f} mm",
            ha="center", va="bottom", fontsize=8, color=color, fontweight="bold",
        )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    smooth = make_smooth_disk(R=80.0)
    toothed = make_toothed_weapon(R_mean=70.0, amplitude=22.0, n_teeth=3)

    print(f"v_per_rad = {V_PER_RAD:.2f} mm/rad  "
          f"(v={V_MS} m/s, ω={RPM} rpm = {OMEGA:.1f} rad/s)")

    contacts_s, r_start_s = analyse_contacts(smooth,  n_spirals=N_SPIRALS,
                                              v_ms=V_MS, rpm=RPM)
    contacts_t, r_start_t = analyse_contacts(toothed, n_spirals=N_SPIRALS,
                                              v_ms=V_MS, rpm=RPM)

    print(f"Smooth  disk  — {len(contacts_s)} contacts, "
          f"bite range [{min(c.bite_depth for c in contacts_s):.1f}, "
          f"{max(c.bite_depth for c in contacts_s):.1f}] mm")
    print(f"3-Tooth weapon — {len(contacts_t)} contacts, "
          f"bite range [{min(c.bite_depth for c in contacts_t):.1f}, "
          f"{max(c.bite_depth for c in contacts_t):.1f}] mm")

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(17, 7))
    gs = fig.add_gridspec(1, 3, width_ratios=[2.2, 2.2, 1.2], wspace=0.30)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    draw_panel(ax1, smooth,  contacts_s, r_start_s,
               "Smooth Disk  (R = 80 mm)")
    draw_panel(ax2, toothed, contacts_t, r_start_t,
               "3-Tooth Weapon  (R_mean = 70 mm, A = 22 mm)")
    draw_distribution(ax3, contacts_s, contacts_t, r_start_s, r_start_t)

    fig.suptitle(
        r"Archimedean Spiral Contact Analysis    "
        rf"$r = r_{{start}} - \frac{{v}}{{\omega}}(\theta - \theta_0)$"
        f"\n"
        f"v = {V_MS} m/s,  ω = {RPM:.0f} rpm = {OMEGA:.1f} rad/s,  "
        f"v/ω = {V_PER_RAD:.1f} mm/rad",
        fontsize=12,
        fontweight="bold",
        y=1.01,
    )

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "spiral_contacts.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\nSaved → {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
