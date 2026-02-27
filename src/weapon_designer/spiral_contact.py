"""Archimedean spiral contact analysis for weapon profiles.

Models an opponent approaching a spinning weapon.  In the weapon's rotating
reference frame the opponent traces an inward Archimedean spiral:

    r_enemy(θ) = r_start − v_per_rad · (θ − θ₀)

where:
    v_per_rad = v_approach_mm_s / omega_rad_s   [mm per radian of weapon rotation]
    θ₀        = starting angle of this spiral in the weapon frame  [rad]

Contact occurs at the first θ ≥ θ₀ where r_enemy(θ) ≤ r_profile(θ).

Why Archimedean?
    For a purely translating opponent (velocity v directed radially inward,
    no rotation) and a weapon rotating at ω, in the weapon frame the opponent
    traces exactly r = r_start − (v/ω)·Δθ.  At 10 m/s approach and 1 000 rpm
    the spiral rate is v/ω ≈ 95.5 mm/rad, meaning one full revolution of the
    weapon sweeps the opponent ~600 mm closer — so first contact typically
    occurs within a fraction of a revolution for any weapon of reasonable size.

Units throughout: mm, rad/s, N (Newton for forces).
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from shapely.geometry import Polygon


# ---------------------------------------------------------------------------
# Data type
# ---------------------------------------------------------------------------

@dataclass
class ContactResult:
    """First-contact data for one spiral trajectory."""
    theta_0: float                        # spiral start angle in weapon frame (rad)
    theta_contact: float                  # angle at first contact (rad)
    r_contact: float                      # radial distance of contact point (mm)
    xy_contact: tuple[float, float]       # Cartesian position of contact (mm)
    outward_normal: tuple[float, float]   # unit outward-surface normal at contact
    force_direction: tuple[float, float]  # unit tangent to spiral path at contact: dP/dθ / |dP/dθ|
    bite_depth: float                     # r_start − r_contact (mm)
    contact_angle_cos: float = 0.0       # |dot(force_direction, outward_normal)|
                                          # 1.0 = perpendicular contact (tooth face-on) → maximum bite
                                          # 0.0 = tangential glancing contact → minimum bite


# ---------------------------------------------------------------------------
# Profile sampling
# ---------------------------------------------------------------------------

def profile_polar(
    poly: "Polygon",
    n_angles: int = 3600,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (angles, radii) for the weapon outer boundary.

    ``angles`` are uniformly spaced in [0, 2π).
    ``radii[i]`` is the maximum distance from the origin at angles[i],
    derived from the exterior ring — handles non-convex profiles correctly
    by taking the outermost boundary point at each angular bin.

    Parameters
    ----------
    poly      : Shapely Polygon of the weapon outer profile (no holes needed).
    n_angles  : number of angular bins; higher = finer contact resolution.
    """
    ext = np.array(poly.exterior.coords)
    x_ext, y_ext = ext[:, 0], ext[:, 1]

    # Dense arc-length interpolation of the exterior ring
    diffs = np.diff(ext, axis=0)
    seg_len = np.hypot(diffs[:, 0], diffs[:, 1])
    arc = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = arc[-1]

    n_dense = max(len(ext) * 4, 4000)
    t_dense = np.linspace(0, total, n_dense, endpoint=False)
    x_d = np.interp(t_dense, arc, x_ext)
    y_d = np.interp(t_dense, arc, y_ext)

    # Convert to polar and wrap to [0, 2π)
    theta_d = np.arctan2(y_d, x_d) % (2.0 * np.pi)
    r_d = np.hypot(x_d, y_d)

    # Build output grid; for each bin keep the maximum r (outermost point)
    dtheta = 2.0 * np.pi / n_angles
    angles = np.linspace(0.0, 2.0 * np.pi, n_angles, endpoint=False)
    radii = np.zeros(n_angles)
    bins = (theta_d / dtheta).astype(int) % n_angles
    np.maximum.at(radii, bins, r_d)

    # Fill any empty bins by linear interpolation
    zero = radii < 1e-3
    if zero.any() and not zero.all():
        nz = np.where(~zero)[0]
        zz = np.where(zero)[0]
        radii[zz] = np.interp(zz, nz, radii[nz])

    return angles, radii


# ---------------------------------------------------------------------------
# Single-spiral contact search
# ---------------------------------------------------------------------------

def find_first_contact(
    angles: np.ndarray,
    radii: np.ndarray,
    v_per_rad: float,
    r_start: float,
    theta_0: float = 0.0,
    n_revolutions: float = 1.5,
) -> ContactResult | None:
    """Find the first contact between one Archimedean spiral and the profile.

    Parameters
    ----------
    angles        : uniform angle grid from ``profile_polar`` (n_angles,)
    radii         : outer radii from ``profile_polar`` (n_angles,)
    v_per_rad     : inward spiral rate (mm per radian of weapon rotation)
    r_start       : spiral starting radius (mm), should be ≥ max profile radius
    theta_0       : spiral origin angle in weapon frame (rad)
    n_revolutions : maximum weapon revolutions to simulate
    """
    n = len(angles)
    dtheta = 2.0 * np.pi / n

    n_steps = int(n_revolutions * n)
    step_arr = np.arange(n_steps, dtype=float)
    theta_arr = theta_0 + step_arr * dtheta
    r_enemy_arr = r_start - v_per_rad * (step_arr * dtheta)

    # Truncate at r = 0
    valid = r_enemy_arr > 0.0
    theta_arr = theta_arr[valid]
    r_enemy_arr = r_enemy_arr[valid]
    if len(theta_arr) == 0:
        return None

    # Profile radius at each step (wrap angle to [0, n))
    idx_arr = (theta_arr / dtheta).astype(int) % n
    r_profile_arr = radii[idx_arr]

    contact_mask = r_enemy_arr <= r_profile_arr
    if not contact_mask.any():
        return None

    first = int(np.argmax(contact_mask))
    theta_c = float(theta_arr[first])
    r_c = float(r_enemy_arr[first])
    xy_c = (r_c * np.cos(theta_c), r_c * np.sin(theta_c))

    # Surface tangent at contact angle via finite difference of profile
    idx_c = int(theta_c / dtheta) % n
    idx_p = (idx_c - 1) % n
    idx_n = (idx_c + 1) % n
    pt_prev = np.array([radii[idx_p] * np.cos(angles[idx_p]),
                        radii[idx_p] * np.sin(angles[idx_p])])
    pt_next = np.array([radii[idx_n] * np.cos(angles[idx_n]),
                        radii[idx_n] * np.sin(angles[idx_n])])
    tangent = pt_next - pt_prev
    t_len = float(np.hypot(tangent[0], tangent[1]))
    if t_len > 1e-9:
        tangent /= t_len
    else:
        tangent = np.array([-np.sin(theta_c), np.cos(theta_c)])

    # Outward normal: tangent rotated 90° CW (for CCW exterior ring)
    outward_normal = (float(tangent[1]), float(-tangent[0]))

    # Force direction: tangent to the spiral path at the contact point.
    # The spiral P(θ) = r(θ)·(cos θ, sin θ) with dr/dθ = −v_per_rad.
    # dP/dθ = (−v_per_rad·cos θ − r·sin θ,  −v_per_rad·sin θ + r·cos θ)
    # This is the velocity direction of the enemy in the weapon frame.
    # By Newton's 3rd law the enemy exerts a force on the weapon in this
    # same direction (impact pushes the weapon along the spiral approach).
    dx = -v_per_rad * np.cos(theta_c) - r_c * np.sin(theta_c)
    dy = -v_per_rad * np.sin(theta_c) + r_c * np.cos(theta_c)
    t_len = np.hypot(dx, dy)
    if t_len > 1e-9:
        force_dir = (float(dx / t_len), float(dy / t_len))
    else:
        force_dir = (float(-np.sin(theta_c)), float(np.cos(theta_c)))

    # Contact angle quality: |dot(spiral_tangent, surface_outward_normal)|
    # High value → spiral approaches face-on (maximum bite energy transfer)
    # Low value  → spiral grazes surface tangentially (glancing, poor bite)
    contact_angle_cos = float(abs(
        force_dir[0] * outward_normal[0] + force_dir[1] * outward_normal[1]
    ))

    return ContactResult(
        theta_0=theta_0,
        theta_contact=theta_c,
        r_contact=r_c,
        xy_contact=xy_c,
        outward_normal=outward_normal,
        force_direction=force_dir,
        bite_depth=r_start - r_c,
        contact_angle_cos=contact_angle_cos,
    )


# ---------------------------------------------------------------------------
# Family analysis
# ---------------------------------------------------------------------------

def analyse_contacts(
    poly: "Polygon",
    n_spirals: int = 20,
    v_ms: float = 10.0,
    rpm: float = 1000.0,
    n_eval: int = 3600,
    n_revolutions: float = 1.5,
) -> tuple[list[ContactResult], float]:
    """Analyse contact for a family of uniformly-spaced Archimedean spirals.

    Parameters
    ----------
    poly          : weapon outer-profile Polygon (no holes)
    n_spirals     : number of spirals; θ₀ uniformly in [0, 2π)
    v_ms          : opponent forward approach speed (m/s)
    rpm           : weapon rotational speed (rpm)
    n_eval        : angular resolution for profile sampling
    n_revolutions : max weapon revolutions per spiral

    Returns
    -------
    contacts  : list of ContactResult (one per spiral that contacts the weapon)
    r_start   : starting radius used for all spirals (mm)
    """
    omega = rpm * 2.0 * np.pi / 60.0         # rad/s
    v_per_rad = (v_ms * 1000.0) / omega       # mm/radian

    angles, radii = profile_polar(poly, n_angles=n_eval)
    r_start = float(radii.max()) * 1.02

    theta_0_values = np.linspace(0.0, 2.0 * np.pi, n_spirals, endpoint=False)

    results: list[ContactResult] = []
    for theta_0 in theta_0_values:
        result = find_first_contact(
            angles, radii, v_per_rad, r_start, float(theta_0), n_revolutions
        )
        if result is not None:
            results.append(result)

    return results, r_start


# ---------------------------------------------------------------------------
# FEA force conversion
# ---------------------------------------------------------------------------

def contact_forces(
    contacts: list[ContactResult],
    force_magnitude_n: float = 1000.0,
    scale_by_angle: bool = True,
) -> list[dict]:
    """Convert a list of ContactResult to point forces for FEA.

    Each force dict contains: ``'x', 'y', 'fx', 'fy'``  (mm and N).

    Force direction is the unit tangent to the Archimedean spiral path at the
    contact point — the velocity direction of the opponent in the weapon's
    rotating frame.  By Newton's 3rd law this is also the direction in which
    the opponent pushes the weapon.

    Spiral path tangent at (r_c, θ_c):
        dP/dθ = (−v_per_rad·cos θ_c − r_c·sin θ_c,
                 −v_per_rad·sin θ_c + r_c·cos θ_c)

    At high RPM (small v_per_rad/r_c) this is nearly tangential to the
    weapon's rotation; at low RPM it has a significant radial component.

    Parameters
    ----------
    contacts          : output from ``analyse_contacts``
    force_magnitude_n : nominal magnitude of each contact force (N)
    scale_by_angle    : if True (default), scale each force by contact_angle_cos
                        so perpendicular (face-on) contacts apply full force
                        while glancing contacts apply proportionally less
    """
    forces = []
    for c in contacts:
        mag = force_magnitude_n * (c.contact_angle_cos if scale_by_angle else 1.0)
        forces.append({
            "x":  c.xy_contact[0],
            "y":  c.xy_contact[1],
            "fx": mag * c.force_direction[0],
            "fy": mag * c.force_direction[1],
            "contact_angle_cos": c.contact_angle_cos,
        })
    return forces
