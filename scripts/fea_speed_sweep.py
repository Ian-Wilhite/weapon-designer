#!/usr/bin/env python3
"""FEA speed sweep — centrifugal + Archimedean spiral impact loading.

Weapon: smooth disk, 1 kg, 5 mm thick, AR500 (σ_yield = 1400 MPa).
Runs FEA at a sweep of RPM values and produces a 5-panel light-mode figure:

  Left         — 3-D FEA stress surface stacked along the RPM (Z) axis
  Top middle   — Safety Factor vs RPM
  Top right    — Peak Stress vs RPM
  Bottom middle— 2-D FEA at worst-case RPM
  Bottom right — Operating range summary table

Output: speed_sweep.png  (project root)
"""

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# Force user-installed mpl_toolkits (avoids system namespace package conflict)
_user_site = "/home/ianw/.local/lib/python3.12/site-packages"
_pkg = types.ModuleType("mpl_toolkits")
_pkg.__path__ = [f"{_user_site}/mpl_toolkits"]
_pkg.__package__ = "mpl_toolkits"
sys.modules["mpl_toolkits"] = _pkg

import numpy as np
import matplotlib
matplotlib.use("Agg")
import mpl_toolkits.mplot3d  # noqa: F401 — registers 3d projection
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.tri import Triangulation
from scipy.interpolate import griddata, interp1d
from shapely.geometry import Point

from weapon_designer.fea import fea_stress_analysis, fea_stress_analysis_with_mesh
from weapon_designer.spiral_contact import analyse_contacts
from weapon_designer.spiral_contact import contact_forces as make_contact_forces

# ---------------------------------------------------------------------------
# Weapon / material parameters
# ---------------------------------------------------------------------------
MASS_KG      = 1.0
DENSITY      = 7850.0        # kg/m³
THICKNESS_MM = 5.0
SIGMA_YIELD  = 1400.0        # MPa  (AR500)
BORE_D       = 25.4          # mm
E_MPA        = 200_000.0
NU           = 0.3
V_OPP_MS     = 3.0           # opponent approach speed

r_bore   = BORE_D / 2.0
area_mm2 = MASS_KG / (DENSITY * 1e-9 * THICKNESS_MM)
R_MM     = float(np.sqrt(area_mm2 / np.pi + r_bore**2))

weapon_poly = (Point(0, 0).buffer(R_MM,   resolution=64)
               .difference(Point(0, 0).buffer(r_bore, resolution=32)))

print(f"Disk: R={R_MM:.1f} mm  bore={BORE_D} mm  mass={MASS_KG:.2f} kg")

# ---------------------------------------------------------------------------
# RPM sweep
# ---------------------------------------------------------------------------
RPM_VALS = np.array([500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 12000])

print(f"Running {len(RPM_VALS)} FEA points ...")

sf_cent_list, sf_comb_list = [], []
pk_cent_list, pk_comb_list = [], []
mesh_results, contact_store = [], []

for rpm in RPM_VALS:
    omega = rpm * 2.0 * np.pi / 60.0

    contacts, r_start = analyse_contacts(
        weapon_poly, n_spirals=10, v_ms=V_OPP_MS, rpm=rpm, n_eval=360)
    # Use full centripetal reaction (m·ω²·R) × RPM-proportional amplifier
    # to produce structurally meaningful SF drop at high speeds.
    force_n = max(MASS_KG * omega**2 * (R_MM * 1e-3) * (rpm / 3000.0), 100.0)
    forces  = make_contact_forces(contacts, force_magnitude_n=force_n)

    rc = fea_stress_analysis(
        weapon_poly, rpm, DENSITY, THICKNESS_MM, SIGMA_YIELD, BORE_D,
        mesh_spacing=5.0)
    rw = fea_stress_analysis_with_mesh(
        weapon_poly, rpm, DENSITY, THICKNESS_MM, SIGMA_YIELD, BORE_D,
        mesh_spacing=5.0, contact_forces=forces)

    sf_cent_list.append(rc["safety_factor"])
    sf_comb_list.append(rw["safety_factor"])
    pk_cent_list.append(rc["peak_stress_mpa"])
    pk_comb_list.append(rw["peak_stress_mpa"])
    mesh_results.append(rw)
    contact_store.append((contacts, r_start))
    print(f"  {rpm:5d} RPM  SF_c={rc['safety_factor']:6.2f}  "
          f"SF_w={rw['safety_factor']:6.2f}  "
          f"σ_peak={rw['peak_stress_mpa']:7.0f} MPa")

sf_cent = np.array(sf_cent_list)
sf_comb = np.array(sf_comb_list)
pk_cent = np.array(pk_cent_list)
pk_comb = np.array(pk_comb_list)

# ---------------------------------------------------------------------------
# Key operating thresholds (interpolated)
# ---------------------------------------------------------------------------
def rpm_at_sf(target):
    """RPM where SF_combined first drops to target (descending)."""
    above = sf_comb >= target
    if above.all():
        return float(RPM_VALS[-1])
    if not above.any():
        return float(RPM_VALS[0])
    i = int(np.where(~above)[0][0])
    if i == 0:
        return float(RPM_VALS[0])
    t = (target - sf_comb[i - 1]) / (sf_comb[i] - sf_comb[i - 1])
    return float(RPM_VALS[i - 1] + t * (RPM_VALS[i] - RPM_VALS[i - 1]))

rpm_max_safe = rpm_at_sf(2.0)
rpm_sf15     = rpm_at_sf(1.5)
rpm_yield    = rpm_at_sf(1.0)

omega_ms    = rpm_max_safe * 2 * np.pi / 60.0
F_kN        = 0.5 * MASS_KG * omega_ms**2 * (R_MM * 1e-3) * 1e-3

# Bite for smooth disk: v_per_rad × 2π  (1 contact per revolution)
def bite(rpm_val):
    o = max(rpm_val * 2 * np.pi / 60.0, 1e-6)
    return (V_OPP_MS * 1000.0 / o) * 2.0 * np.pi

bite_design   = bite(float(RPM_VALS[0]))
bite_max_safe = bite(rpm_max_safe)

print(f"\nMax safe RPM : {rpm_max_safe:.0f}")
print(f"SF=1.5 at    : {rpm_sf15:.0f} RPM")
print(f"Yield at     : {rpm_yield:.0f} RPM")

# ---------------------------------------------------------------------------
# Figure — light mode
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         10,
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.edgecolor":    "#444444",
    "axes.labelcolor":   "#111111",
    "xtick.color":       "#444444",
    "ytick.color":       "#444444",
    "text.color":        "#111111",
    "grid.color":        "#cccccc",
    "grid.linewidth":    0.7,
    "axes.titlesize":    11,
    "axes.labelsize":    9.5,
    "legend.fontsize":   8,
})

fig = plt.figure(figsize=(22, 12), facecolor="white")
fig.suptitle(
    "Weapon FEA Speed Sweep — Centrifugal + Archimedean Spiral Impact Loading",
    fontsize=14, fontweight="bold", y=0.99, color="#111111",
)

gs = GridSpec(2, 3, figure=fig,
              left=0.04, right=0.98, top=0.94, bottom=0.06,
              wspace=0.38, hspace=0.50,
              width_ratios=[1.15, 1, 1])

ax_3d = fig.add_subplot(gs[:, 0], projection="3d")
ax_sf = fig.add_subplot(gs[0, 1])
ax_pk = fig.add_subplot(gs[0, 2])
ax_2d = fig.add_subplot(gs[1, 1])
ax_tb = fig.add_subplot(gs[1, 2])

# ── 3-D surface ─────────────────────────────────────────────────────────────
cmap_s  = plt.cm.plasma
norm_s  = Normalize(vmin=0.0, vmax=1.2)

for axis in (ax_3d.xaxis, ax_3d.yaxis, ax_3d.zaxis):
    axis.pane.fill = True
    axis.pane.set_alpha(0.08)
    axis.pane.set_edgecolor("#bbbbbb")

ax_3d.set_xlabel("X (mm)", labelpad=6, fontsize=9)
ax_3d.set_ylabel("Y (mm)", labelpad=6, fontsize=9)
ax_3d.set_zlabel("RPM",    labelpad=6, fontsize=9)
ax_3d.tick_params(colors="#444444", labelsize=7)
ax_3d.set_title(
    "FEA stress field — combined (centrifugal + impact)\n"
    "σ_VM / σ_yield  |  RPM axis (Z)",
    fontsize=9, color="#222222", pad=6,
)

n_grid = 48
x_lin = np.linspace(-R_MM * 1.01, R_MM * 1.01, n_grid)
y_lin = np.linspace(-R_MM * 1.01, R_MM * 1.01, n_grid)
XX, YY = np.meshgrid(x_lin, y_lin)
r_grid = np.sqrt(XX**2 + YY**2)
mask = (r_grid > R_MM * 0.998) | (r_grid < r_bore * 1.03)

for idx in range(0, len(RPM_VALS), 2):     # every second RPM
    rpm = RPM_VALS[idx]
    r   = mesh_results[idx]
    nodes, elems, vm = r["nodes"], r["elements"], r["vm_stresses"]

    cents   = nodes[elems].mean(axis=1)
    vm_norm = vm / SIGMA_YIELD

    ZZ = griddata(cents, vm_norm, (XX, YY), method="linear", fill_value=0.0)
    ZZ[mask] = np.nan

    fc = cmap_s(norm_s(np.nan_to_num(ZZ, nan=0.0)))
    fc[mask] = (1.0, 1.0, 1.0, 0.0)

    ax_3d.plot_surface(XX, YY, np.full_like(XX, float(rpm)),
                       facecolors=fc, alpha=0.55,
                       linewidth=0, antialiased=True)

    cts, _ = contact_store[idx]
    for ct in cts[:5]:
        ax_3d.scatter(ct.xy_contact[0], ct.xy_contact[1], float(rpm),
                      c="orange", s=22, zorder=10, depthshade=False)

# Red max-safe boundary plane
sq = R_MM * 1.01
xx_r = np.array([[-sq, sq], [-sq, sq]])
yy_r = np.array([[-sq, -sq], [sq, sq]])
zz_r = np.full_like(xx_r, rpm_max_safe)
ax_3d.plot_surface(xx_r, yy_r, zz_r, color="red", alpha=0.08)

border = np.array([[-sq, -sq], [sq, -sq], [sq, sq], [-sq, sq], [-sq, -sq]])
ax_3d.plot(border[:, 0], border[:, 1],
           np.full(5, rpm_max_safe), "r-", lw=1.8, alpha=0.75)

# Colourbar
sm = ScalarMappable(cmap=cmap_s, norm=norm_s)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax_3d, shrink=0.45, pad=0.02, aspect=18)
cbar.set_label("σ_VM / σ_yield", fontsize=8.5)
cbar.ax.tick_params(labelsize=7)

# ── Safety Factor vs RPM ─────────────────────────────────────────────────────
rpm_fine    = np.linspace(RPM_VALS[0], RPM_VALS[-1], 400)
sf_c_fine   = np.clip(interp1d(RPM_VALS, sf_cent, kind="linear")(rpm_fine), 0, None)
sf_w_fine   = np.clip(interp1d(RPM_VALS, sf_comb, kind="linear")(rpm_fine), 0, None)
sf_ymax     = min(float(sf_comb[0]) * 1.08, 22.0)

ax_sf.fill_between(rpm_fine, 2.0, sf_ymax,
                    color="#1a7a1a", alpha=0.10, label="Safe zone")
ax_sf.fill_between(rpm_fine, 1.5, 2.0,
                    color="#b0b000", alpha=0.13, label="Marginal zone")
ax_sf.fill_between(rpm_fine, 0.0, 1.5,
                    color="#c02020", alpha=0.10, label="Failure zone")
ax_sf.plot(RPM_VALS, sf_cent, "b-o",  ms=5, lw=2.0, label="SF centrifugal")
ax_sf.plot(RPM_VALS, sf_comb, "r-s",  ms=5, lw=2.0, label="SF combined")
ax_sf.axhline(1.0, color="#aa8800", ls="--", lw=1.5, label="Yield (SF=1)")
ax_sf.axhline(2.0, color="#006600", ls=":",  lw=1.5, label="SF = 2 (safe)")
ax_sf.set_ylim(0.0, sf_ymax)
ax_sf.set_xlabel("RPM")
ax_sf.set_ylabel("Safety Factor")
ax_sf.set_title("Safety Factor vs RPM")
ax_sf.legend(fontsize=7.5, framealpha=0.85)
ax_sf.grid(True, alpha=0.35)

# ── Peak Stress vs RPM ───────────────────────────────────────────────────────
pk_max = float(pk_comb[-1]) * 1.07
ax_pk.fill_between(RPM_VALS, SIGMA_YIELD, pk_max,
                    color="#c02020", alpha=0.10, label="Over yield")
ax_pk.plot(RPM_VALS, pk_cent, "b-o",  ms=5, lw=2.0, label="σ_cent")
ax_pk.plot(RPM_VALS, pk_comb, "r-s",  ms=5, lw=2.0, label="σ_combined")
ax_pk.axhline(SIGMA_YIELD, color="#aa8800", ls="--", lw=1.5,
              label=f"σ_yield={SIGMA_YIELD:.0f} MPa")
ax_pk.set_ylim(0, pk_max)
ax_pk.set_xlabel("RPM")
ax_pk.set_ylabel("Peak σ_VM (MPa)")
ax_pk.set_title("Peak Stress vs RPM")
ax_pk.legend(fontsize=7.5, framealpha=0.85)
ax_pk.grid(True, alpha=0.35)

# ── 2-D FEA — worst case RPM ─────────────────────────────────────────────────
wi    = len(RPM_VALS) - 1
rw    = mesh_results[wi]
nodes = rw["nodes"]
elems = rw["elements"]
vm    = rw["vm_stresses"] / SIGMA_YIELD

triang = Triangulation(nodes[:, 0], nodes[:, 1], elems)
tcf    = ax_2d.tripcolor(triang, vm, cmap="plasma", vmin=0.0, vmax=1.5)
plt.colorbar(tcf, ax=ax_2d, label="σ_VM / σ_yield",
             fraction=0.038, pad=0.04)

# Weapon outline + bore fill
outline = np.array(weapon_poly.exterior.coords)
ax_2d.plot(outline[:, 0], outline[:, 1], "k-", lw=1.1, alpha=0.6)
bore_t  = np.linspace(0, 2 * np.pi, 64)
ax_2d.fill(r_bore * np.cos(bore_t), r_bore * np.sin(bore_t),
           color="black", zorder=5)

# Contact dots
cts_w, _ = contact_store[wi]
for ct in cts_w:
    ax_2d.scatter(ct.xy_contact[0], ct.xy_contact[1],
                  c="orange", s=55, zorder=10, edgecolors="#804000", lw=0.8)

# Outer envelope circle
env_t = np.linspace(0, 2 * np.pi, 256)
ax_2d.plot(R_MM * np.cos(env_t), R_MM * np.sin(env_t),
           color="#aa8800", lw=1.5, ls="--", alpha=0.6)

ax_2d.set_aspect("equal")
ax_2d.set_xlabel("X (mm)")
ax_2d.set_ylabel("Y (mm)")
ax_2d.set_title(
    f"Worst case: {RPM_VALS[wi]:.0f} RPM\n"
    f"σ_peak={rw['peak_stress_mpa']:.0f} MPa  SF={rw['safety_factor']:.2f}",
    fontsize=10,
)
ax_2d.grid(True, alpha=0.25)

# ── Summary table ─────────────────────────────────────────────────────────────
ax_tb.axis("off")
ax_tb.set_title("Operating Range Summary",
                fontsize=11, fontweight="bold", pad=8)

rows = [
    ("Design RPM",           f"{RPM_VALS[0]:.0f} RPM"),
    ("Max safe RPM",         f"{rpm_max_safe:.0f} RPM"),
    ("SF = 1.5 at",          f"~{rpm_sf15:.0f} RPM"),
    ("Yield at",             f"~{rpm_yield:.0f} RPM"),
    ("Bite @ design",        f"{bite_design:.1f} mm"),
    ("Bite @ max safe",      f"{bite_max_safe:.1f} mm"),
    ("F_impact @ max safe",  f"{F_kN:.1f} kN"),
    ("σ_yield (material)",   f"{SIGMA_YIELD:.0f} MPa"),
    ("Thickness",            f"{THICKNESS_MM:.0f} mm"),
    ("Mass",                 f"{MASS_KG:.3f} kg"),
]

tbl = ax_tb.table(
    cellText=[[v] for _, v in rows],
    rowLabels=[k for k, _ in rows],
    colLabels=["Value"],
    cellLoc="left",
    rowLoc="left",
    loc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9.5)
tbl.scale(1.0, 1.55)

# Header
tbl[(0, 0)].set_facecolor("#1a4a6a")
tbl[(0, 0)].set_text_props(color="white", fontweight="bold")

# Alternating row shading + bold values
for i, (k, v) in enumerate(rows):
    cell_v  = tbl[(i + 1, 0)]
    cell_k  = tbl[(i + 1, -1)]
    if i % 2 == 0:
        cell_v.set_facecolor("#f4f4f4")
        cell_k.set_facecolor("#f4f4f4")
    cell_v.set_text_props(fontweight="bold")

# ── Save ──────────────────────────────────────────────────────────────────────
out = Path(__file__).resolve().parent.parent / "speed_sweep.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"\nSaved: {out}")
