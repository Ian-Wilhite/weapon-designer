"""Generate weapon archetype silhouette panels for the poster Introduction.

Produces poster/assets/weapon_archetypes.png — 4 panels showing the four
design test cases as clean top-down weapon silhouettes with weight-relief
pockets, bore hole, and rotation arrow.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from shapely.geometry import Polygon

from weapon_designer.profile_splines import build_catmull_rom_profile


# ── Color palette ─────────────────────────────────────────────────────────────
COLORS = {
    'fw':   '#3b6ea5',   # steel blue  — featherweight disk
    'bar':  '#2d6a4f',   # forest green — compact bar
    'egg':  '#7b2d8b',   # purple       — eggbeater
    'hw':   '#7d4f00',   # bronze       — heavyweight disk
}
EDGE_C  = '#1a1a2e'
WHITE   = '#ffffff'
ROT_C   = '#c0392b'
BGPANEL = '#f0f2f5'

plt.rcParams.update({'font.family': 'DejaVu Sans', 'figure.facecolor': 'white'})


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ellipse_polygon(cx, cy, rx, ry, angle_deg, n=56):
    """Shapely polygon for an ellipse (possibly rotated)."""
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    a  = np.radians(angle_deg)
    lx = rx * np.cos(th)
    ly = ry * np.sin(th)
    ex = cx + lx * np.cos(a) - ly * np.sin(a)
    ey = cy + lx * np.sin(a) + ly * np.cos(a)
    p  = Polygon(zip(ex, ey))
    return p.buffer(0) if not p.is_valid else p


def _draw_poly(ax, poly, facecolor, edgecolor, lw=1.4, zorder=2):
    if poly is None or poly.is_empty:
        return
    geoms = [poly] if poly.geom_type == 'Polygon' else list(poly.geoms)
    for g in geoms:
        x, y = g.exterior.xy
        ax.fill(x, y, color=facecolor, zorder=zorder, linewidth=0)
        ax.plot(x, y, color=edgecolor, lw=lw, zorder=zorder + 1)
        for interior in g.interiors:
            ix, iy = interior.xy
            ax.fill(ix, iy, color=WHITE, zorder=zorder + 2, linewidth=0)
            ax.plot(ix, iy, color=edgecolor, lw=lw * 0.7, zorder=zorder + 3)


def draw_weapon_panel(ax, radii, max_r, bore_d, pocket_specs,
                      color, title, specs_lines):
    """Render one weapon archetype panel.

    Parameters
    ----------
    pocket_specs : list of (cx, cy, rx, ry, angle_deg)
    """
    bore_r = bore_d / 2.0
    N      = len(radii)

    # Build outer profile via Catmull-Rom (interpolates every control point)
    min_r_param = bore_r * 0.85
    poly = build_catmull_rom_profile(
        np.asarray(radii, dtype=float), max_r, min_r_param, n_eval=720, alpha=0.5
    )
    if poly is None or poly.is_empty:
        # Fallback: piecewise-linear
        theta_c = np.linspace(0, 2 * np.pi, N, endpoint=False)
        theta_e = np.linspace(0, 2 * np.pi, 720, endpoint=False)
        r_e = np.interp(
            theta_e,
            np.append(theta_c, 2 * np.pi),
            np.append(radii, radii[0])
        )
        poly = Polygon(zip(r_e * np.cos(theta_e), r_e * np.sin(theta_e))).buffer(0)

    # Subtract weight-relief pockets
    body = poly
    for (cx, cy, rx, ry, ang) in pocket_specs:
        pocket = _ellipse_polygon(cx, cy, rx, ry, ang)
        if pocket.is_valid and not pocket.is_empty:
            try:
                body = body.difference(pocket)
            except Exception:
                pass

    # ── Draw weapon body ──────────────────────────────────────────────────────
    ax.set_facecolor(BGPANEL)
    for sp in ax.spines.values():
        sp.set_visible(False)

    _draw_poly(ax, body, color, EDGE_C, lw=1.4)

    # ── Bore hole ─────────────────────────────────────────────────────────────
    theta = np.linspace(0, 2 * np.pi, 240)
    bx = bore_r * np.cos(theta)
    by = bore_r * np.sin(theta)
    ax.fill(bx, by, color=WHITE, zorder=8, linewidth=0)
    ax.plot(bx, by, color=EDGE_C, lw=1.0, zorder=9)
    # Center cross
    cr = bore_r * 0.55
    ax.plot([0, 0], [-cr, cr], color='#888888', lw=0.7, zorder=10)
    ax.plot([-cr, cr], [0, 0], color='#888888', lw=0.7, zorder=10)

    # ── Rotation arc + arrow ──────────────────────────────────────────────────
    arc_r  = max_r * 1.18
    arc_th = np.linspace(0.10 * np.pi, 0.48 * np.pi, 30)
    ax.plot(arc_r * np.cos(arc_th), arc_r * np.sin(arc_th),
            color=ROT_C, lw=2.2, zorder=11)
    # Arrowhead via annotation
    i2, i1 = -1, -4
    ax.annotate(
        '',
        xy=(arc_r * np.cos(arc_th[i2]), arc_r * np.sin(arc_th[i2])),
        xytext=(arc_r * np.cos(arc_th[i1]), arc_r * np.sin(arc_th[i1])),
        arrowprops=dict(arrowstyle='->', color=ROT_C, lw=1.8, mutation_scale=12),
        zorder=12,
    )

    # ── Axes limits & formatting ──────────────────────────────────────────────
    lim = max_r * 1.38
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim * 0.95, lim * 1.28)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_title(title, fontsize=11.5, fontweight='bold', color=color, pad=5)
    for i, line in enumerate(specs_lines):
        ax.text(0.5, -0.03 - i * 0.095, line,
                transform=ax.transAxes, ha='center', va='top',
                fontsize=9.0, color='#333333')


# ── Weapon definitions ────────────────────────────────────────────────────────

N     = 12
theta = np.linspace(0, 2 * np.pi, N, endpoint=False)

# 1. Featherweight disk — bilobal, two strike surfaces
r_fw      = 76.0 + 10.0 * np.cos(2.0 * theta)
pockets_fw = [
    ( 50.0,  0.0, 16.0, 9.0,   0.0),
    (-50.0,  0.0, 16.0, 9.0,   0.0),
]

# 2. Compact bar — strongly bilobal, distinct bar shape
r_bar      = 62.0 + 52.0 * np.cos(2.0 * theta)
pockets_bar = [
    ( 74.0,  0.0, 26.0, 11.0,  0.0),
    (-74.0,  0.0, 26.0, 11.0,  0.0),
]

# 3. Eggbeater — three blades at 120° spacing
r_egg      = 87.0 + 56.0 * np.cos(3.0 * theta)
# Pocket centers at r=108mm along each blade axis
_r_ep = 108.0
pockets_egg = [
    (_r_ep * np.cos(0.0),           _r_ep * np.sin(0.0),           22.0, 10.0,   0.0),
    (_r_ep * np.cos(2*np.pi/3),     _r_ep * np.sin(2*np.pi/3),     22.0, 10.0, 120.0),
    (_r_ep * np.cos(4*np.pi/3),     _r_ep * np.sin(4*np.pi/3),     22.0, 10.0, 240.0),
]

# 4. Heavyweight disk — large, near-circular, with 6 weight-relief pockets
r_hw       = 172.0 + 5.0 * np.cos(2.0 * theta) + 3.0 * np.cos(4.0 * theta)
_r_hp = 120.0
pockets_hw = [
    (_r_hp * np.cos(k * np.pi / 3), _r_hp * np.sin(k * np.pi / 3),
     24.0, 16.0, np.degrees(k * np.pi / 3))
    for k in range(6)
]


# ── Figure ────────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(13, 13), dpi=150)
gs  = GridSpec(2, 2, figure=fig,
               left=0.01, right=0.99, top=0.93, bottom=0.05,
               wspace=0.10, hspace=0.20)

draw_weapon_panel(
    fig.add_subplot(gs[0, 0]),
    r_fw, 80.0, 25.4, pockets_fw,
    COLORS['fw'],
    'Featherweight Disk',
    ['0.75 kg · 12,000 RPM', 'ø160 mm · S7 Tool Steel', r'$E_k \approx 3{,}200$ J'],
)

draw_weapon_panel(
    fig.add_subplot(gs[0, 1]),
    r_bar, 120.0, 25.4, pockets_bar,
    COLORS['bar'],
    'Compact Bar',
    ['1.00 kg · 8,000 RPM', 'ø240 mm · S7 Tool Steel', r'$E_k \approx 3{,}500$ J'],
)

draw_weapon_panel(
    fig.add_subplot(gs[1, 0]),
    r_egg, 150.0, 31.75, pockets_egg,
    COLORS['egg'],
    'Eggbeater (3-blade)',
    ['2.50 kg · 8,000 RPM', 'ø300 mm · S7 Tool Steel', r'$E_k \approx 12{,}000$ J'],
)

draw_weapon_panel(
    fig.add_subplot(gs[1, 1]),
    r_hw, 180.0, 38.1, pockets_hw,
    COLORS['hw'],
    'Heavyweight Disk',
    ['7.00 kg · 8,000 RPM', 'ø360 mm · AR500 Steel', r'$E_k \approx 40{,}000$ J'],
)

fig.suptitle(
    'Design Test Cases — Combat Spinner Weapon Archetypes',
    fontsize=13.5, fontweight='bold', y=0.97, color=EDGE_C,
)

out_path = os.path.join(os.path.dirname(__file__), 'assets', 'weapon_archetypes.png')
fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f'Saved: {out_path}')
plt.close(fig)
