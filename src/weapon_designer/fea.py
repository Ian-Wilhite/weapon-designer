"""Lightweight 2D plane-stress FEA for spinning weapon structural analysis.

Uses constant-strain triangular (CST) elements with centrifugal body loading.
This is an approximation suitable for rapid design iteration — not a
replacement for full FEA validation.

All units: mm, kg, N, MPa (N/mm²).
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from shapely.geometry import Polygon, MultiPolygon


# ---------------------------------------------------------------------------
# Meshing
# ---------------------------------------------------------------------------

def _triangulate_polygon(poly: Polygon, max_area: float = 50.0) -> tuple[np.ndarray, np.ndarray]:
    """Triangulate a Shapely polygon using ear-clipping via scipy Delaunay.

    Returns (nodes, elements) where nodes is (N, 2) and elements is (M, 3).
    Points inside holes are excluded.
    """
    from scipy.spatial import Delaunay

    # Sample boundary points
    boundary_pts = _sample_boundary(poly, spacing=3.0)

    # Add interior grid points for mesh density
    interior_pts = _interior_grid(poly, spacing=max_area ** 0.5)

    all_pts = np.vstack([boundary_pts, interior_pts]) if len(interior_pts) > 0 else boundary_pts

    if len(all_pts) < 3:
        return np.zeros((0, 2)), np.zeros((0, 3), dtype=int)

    # Delaunay triangulation
    tri = Delaunay(all_pts)
    simplices = tri.simplices

    # Filter triangles: keep only those whose centroid is inside the polygon
    centroids = all_pts[simplices].mean(axis=1)
    from shapely.geometry import Point
    mask = np.array([poly.contains(Point(c[0], c[1])) for c in centroids])
    elements = simplices[mask]

    if len(elements) == 0:
        return all_pts, np.zeros((0, 3), dtype=int)

    return all_pts, elements


def _sample_boundary(poly: Polygon, spacing: float = 3.0) -> np.ndarray:
    """Sample points along exterior and interior boundaries."""
    points = []

    def _sample_ring(coords, sp):
        ring_pts = np.array(coords[:-1])  # drop closing duplicate
        # Interpolate to roughly uniform spacing
        if len(ring_pts) < 2:
            return ring_pts
        diffs = np.diff(ring_pts, axis=0)
        lengths = np.hypot(diffs[:, 0], diffs[:, 1])
        total = lengths.sum()
        n_pts = max(int(total / sp), len(ring_pts))
        # Subsample by arc-length
        cum = np.concatenate([[0], np.cumsum(lengths)])
        targets = np.linspace(0, total, n_pts, endpoint=False)
        sampled = np.zeros((n_pts, 2))
        for i, t in enumerate(targets):
            idx = np.searchsorted(cum, t, side='right') - 1
            idx = min(idx, len(ring_pts) - 2)
            frac = (t - cum[idx]) / max(lengths[idx], 1e-12)
            sampled[i] = ring_pts[idx] + frac * diffs[idx]
        return sampled

    if isinstance(poly, MultiPolygon):
        for p in poly.geoms:
            points.append(_sample_boundary(p, spacing))
        return np.vstack(points) if points else np.zeros((0, 2))

    points.append(_sample_ring(poly.exterior.coords, spacing))
    for interior in poly.interiors:
        points.append(_sample_ring(interior.coords, spacing))

    return np.vstack(points)


def _interior_grid(poly: Polygon, spacing: float = 5.0) -> np.ndarray:
    """Generate a grid of interior points within the polygon."""
    if isinstance(poly, MultiPolygon):
        grids = [_interior_grid(p, spacing) for p in poly.geoms]
        return np.vstack(grids) if grids else np.zeros((0, 2))

    bounds = poly.bounds  # (minx, miny, maxx, maxy)
    xs = np.arange(bounds[0] + spacing, bounds[2], spacing)
    ys = np.arange(bounds[1] + spacing, bounds[3], spacing)
    grid = np.array([(x, y) for x in xs for y in ys])

    if len(grid) == 0:
        return np.zeros((0, 2))

    from shapely.geometry import Point
    mask = np.array([poly.contains(Point(p[0], p[1])) for p in grid])
    return grid[mask]


# ---------------------------------------------------------------------------
# CST Element Stiffness
# ---------------------------------------------------------------------------

def _cst_stiffness(
    nodes: np.ndarray,
    elem: np.ndarray,
    E: float,
    nu: float,
    thickness: float,
) -> np.ndarray:
    """Compute 6x6 stiffness matrix for a constant-strain triangle.

    Plane stress formulation.
    nodes: (3, 2) coordinates of triangle vertices
    """
    x = nodes[:, 0]
    y = nodes[:, 1]

    # Area via cross product
    x0, x1, x2 = x
    y0, y1, y2 = y
    A2 = abs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0))
    area = A2 / 2.0

    if area < 1e-12:
        return np.zeros((6, 6))

    # B matrix (strain-displacement)
    b = np.array([y1 - y2, y2 - y0, y0 - y1])
    c = np.array([x2 - x1, x0 - x2, x1 - x0])

    B = np.zeros((3, 6))
    for i in range(3):
        B[0, 2 * i] = b[i]
        B[1, 2 * i + 1] = c[i]
        B[2, 2 * i] = c[i]
        B[2, 2 * i + 1] = b[i]
    B /= A2

    # D matrix (plane stress constitutive)
    D = (E / (1 - nu ** 2)) * np.array([
        [1, nu, 0],
        [nu, 1, 0],
        [0, 0, (1 - nu) / 2],
    ])

    # K = t * A * B^T D B
    Ke = thickness * area * (B.T @ D @ B)
    return Ke


# ---------------------------------------------------------------------------
# Assembly and solve
# ---------------------------------------------------------------------------

def _assemble_global(
    nodes: np.ndarray,
    elements: np.ndarray,
    E: float,
    nu: float,
    thickness: float,
) -> sparse.csr_matrix:
    """Assemble global stiffness matrix from CST elements."""
    n_nodes = len(nodes)
    n_dof = 2 * n_nodes

    rows, cols, vals = [], [], []

    for elem in elements:
        tri_nodes = nodes[elem]
        Ke = _cst_stiffness(tri_nodes, elem, E, nu, thickness)

        # DOF mapping: node i -> dofs [2i, 2i+1]
        dofs = np.array([2 * elem[0], 2 * elem[0] + 1,
                         2 * elem[1], 2 * elem[1] + 1,
                         2 * elem[2], 2 * elem[2] + 1])

        for i in range(6):
            for j in range(6):
                if abs(Ke[i, j]) > 1e-20:
                    rows.append(dofs[i])
                    cols.append(dofs[j])
                    vals.append(Ke[i, j])

    K = sparse.coo_matrix((vals, (rows, cols)), shape=(n_dof, n_dof))
    return K.tocsr()


def _find_boundary_edges(elements: np.ndarray) -> np.ndarray:
    """Find boundary edges of the triangulated mesh.

    Boundary edges are shared by exactly one triangle.

    Parameters
    ----------
    elements : (M, 3) int array of triangle vertex indices

    Returns
    -------
    boundary_edges : (B, 2) int array of node-index pairs (sorted per edge)
    """
    from collections import Counter

    edge_count: Counter = Counter()
    for tri in elements:
        for i, j in ((0, 1), (1, 2), (2, 0)):
            edge = (int(min(tri[i], tri[j])), int(max(tri[i], tri[j])))
            edge_count[edge] += 1

    boundary = [list(e) for e, cnt in edge_count.items() if cnt == 1]
    if boundary:
        return np.array(boundary, dtype=int)
    return np.zeros((0, 2), dtype=int)


def _apply_contact_forces(
    F: np.ndarray,
    nodes: np.ndarray,
    contact_forces: list[dict],
    mode: str = "neumann_edge",
    elements: np.ndarray | None = None,
) -> np.ndarray:
    """Add point contact forces to the global force vector F (in-place).

    Each entry in ``contact_forces`` must be a dict with keys:
        'x', 'y'   : position of force application (mm)
        'fx', 'fy' : force components (N)

    Parameters
    ----------
    mode : "neumann_edge" (default) or "nearest_node"
        "nearest_node" — legacy KDTree nearest-node injection on all nodes.
        "neumann_edge" — finds the two nearest boundary nodes, projects the
        contact point onto the edge between them, and distributes force with
        linear weight t.  Requires ``elements`` to extract boundary edges.
        Falls back to "nearest_node" when boundary set is empty.
    elements : (M, 3) int array, required for "neumann_edge" mode.
    """
    from scipy.spatial import KDTree

    if not contact_forces:
        return F

    if mode == "nearest_node":
        tree = KDTree(nodes)
        for cf in contact_forces:
            pt = np.array([cf["x"], cf["y"]])
            _, idx = tree.query(pt)
            F[2 * idx] += cf["fx"]
            F[2 * idx + 1] += cf["fy"]
        return F

    # --- Neumann-edge mode ---
    boundary_edges = np.zeros((0, 2), dtype=int)
    if elements is not None and len(elements) > 0:
        boundary_edges = _find_boundary_edges(elements)

    if len(boundary_edges) == 0:
        # Fallback: nearest node on all nodes
        tree = KDTree(nodes)
        for cf in contact_forces:
            pt = np.array([cf["x"], cf["y"]])
            _, idx = tree.query(pt)
            F[2 * idx] += cf["fx"]
            F[2 * idx + 1] += cf["fy"]
        return F

    # Build KDTree on boundary node positions only
    boundary_node_set = np.unique(boundary_edges.ravel())
    boundary_positions = nodes[boundary_node_set]
    tree = KDTree(boundary_positions)

    for cf in contact_forces:
        pt = np.array([cf["x"], cf["y"]])
        fx, fy = cf["fx"], cf["fy"]

        k = min(2, len(boundary_node_set))
        _, local_indices = tree.query(pt, k=k)
        local_indices = np.atleast_1d(local_indices)

        ni = boundary_node_set[local_indices[0]]
        if k < 2 or local_indices[0] == local_indices[-1]:
            # Only one distinct node found
            F[2 * ni] += fx
            F[2 * ni + 1] += fy
            continue

        nj = boundary_node_set[local_indices[1]]
        if ni == nj:
            F[2 * ni] += fx
            F[2 * ni + 1] += fy
            continue

        # Project pt onto segment ni→nj, weight t ∈ [0, 1]
        pi = nodes[ni]
        pj = nodes[nj]
        seg = pj - pi
        seg_len_sq = float(np.dot(seg, seg))
        if seg_len_sq < 1e-12:
            F[2 * ni] += fx
            F[2 * ni + 1] += fy
            continue

        t = float(np.clip(np.dot(pt - pi, seg) / seg_len_sq, 0.0, 1.0))
        F[2 * ni] += (1.0 - t) * fx
        F[2 * ni + 1] += (1.0 - t) * fy
        F[2 * nj] += t * fx
        F[2 * nj + 1] += t * fy

    return F


def _centrifugal_load(
    nodes: np.ndarray,
    elements: np.ndarray,
    omega_rad_s: float,
    density_tonne_mm3: float,
    thickness: float,
) -> np.ndarray:
    """Compute consistent centrifugal body force vector.

    Uses mm-N-MPa-tonne unit system:
      f_body = rho [tonne/mm³] * omega² [1/s²] * r [mm] → N/mm³

    Forces are distributed to element nodes (1/3 of element force per node).
    """
    n_dof = 2 * len(nodes)
    F = np.zeros(n_dof)

    for elem in elements:
        tri_nodes = nodes[elem]
        x = tri_nodes[:, 0]
        y = tri_nodes[:, 1]

        # Element area
        area = 0.5 * abs((x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0]))
        if area < 1e-12:
            continue

        # Centrifugal force at centroid
        cx = tri_nodes[:, 0].mean()
        cy = tri_nodes[:, 1].mean()
        r = np.hypot(cx, cy)

        if r < 1e-6:
            continue

        # Force magnitude per unit volume: rho * omega^2 * r
        # Total element force: rho * omega^2 * r * area * thickness
        f_mag = density_tonne_mm3 * omega_rad_s ** 2 * r * area * thickness

        # Direction: radially outward from origin
        fx = f_mag * cx / r
        fy = f_mag * cy / r

        # Distribute equally to 3 nodes
        for i in range(3):
            node_id = elem[i]
            F[2 * node_id] += fx / 3.0
            F[2 * node_id + 1] += fy / 3.0

    return F


def _apply_boundary_conditions(
    K: sparse.csr_matrix,
    F: np.ndarray,
    nodes: np.ndarray,
    bore_radius: float,
) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
    """Pin nodes near the bore (hub) to simulate shaft constraint.

    Returns (K_modified, F_modified, free_dofs).
    """
    n_dof = len(F)
    r = np.hypot(nodes[:, 0], nodes[:, 1])
    # Pin nodes within 1.5x bore radius
    fixed_nodes = np.where(r <= bore_radius * 1.5)[0]

    if len(fixed_nodes) == 0:
        # Fallback: pin the 3 nodes closest to origin
        closest = np.argsort(r)[:3]
        fixed_nodes = closest

    fixed_dofs = np.sort(np.concatenate([2 * fixed_nodes, 2 * fixed_nodes + 1]))
    free_dofs = np.setdiff1d(np.arange(n_dof), fixed_dofs)

    return K, F, free_dofs


def _compute_von_mises(
    nodes: np.ndarray,
    elements: np.ndarray,
    displacements: np.ndarray,
    E: float,
    nu: float,
) -> np.ndarray:
    """Compute von Mises stress for each element.

    Returns array of shape (n_elements,) in MPa (N/mm²).
    """
    D = (E / (1 - nu ** 2)) * np.array([
        [1, nu, 0],
        [nu, 1, 0],
        [0, 0, (1 - nu) / 2],
    ])

    stresses = np.zeros(len(elements))

    for idx, elem in enumerate(elements):
        tri_nodes = nodes[elem]
        x = tri_nodes[:, 0]
        y = tri_nodes[:, 1]

        x0, x1, x2 = x
        y0, y1, y2 = y
        A2 = abs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0))

        if A2 < 1e-12:
            continue

        b = np.array([y1 - y2, y2 - y0, y0 - y1])
        c = np.array([x2 - x1, x0 - x2, x1 - x0])

        B = np.zeros((3, 6))
        for i in range(3):
            B[0, 2 * i] = b[i]
            B[1, 2 * i + 1] = c[i]
            B[2, 2 * i] = c[i]
            B[2, 2 * i + 1] = b[i]
        B /= A2

        # Element displacements
        dofs = np.array([2 * elem[0], 2 * elem[0] + 1,
                         2 * elem[1], 2 * elem[1] + 1,
                         2 * elem[2], 2 * elem[2] + 1])
        u_e = displacements[dofs]

        # Stress: sigma = D @ B @ u
        strain = B @ u_e
        stress = D @ strain  # [sigma_x, sigma_y, tau_xy]

        sx, sy, txy = stress
        # von Mises for plane stress
        vm = np.sqrt(sx ** 2 - sx * sy + sy ** 2 + 3 * txy ** 2)
        stresses[idx] = vm

    return stresses


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fea_stress_analysis(
    poly: Polygon | MultiPolygon,
    rpm: float,
    density_kg_m3: float,
    thickness_mm: float,
    yield_strength_mpa: float,
    bore_diameter_mm: float,
    youngs_modulus_mpa: float = 200_000.0,
    poissons_ratio: float = 0.3,
    mesh_spacing: float = 5.0,
    contact_forces: list[dict] | None = None,
    contact_load_mode: str = "neumann_edge",
) -> dict:
    """Run 2D plane-stress FEA under centrifugal loading.

    Parameters
    ----------
    poly : weapon polygon (mm units)
    rpm : spin speed
    density_kg_m3 : material density
    thickness_mm : sheet thickness
    yield_strength_mpa : material yield strength
    bore_diameter_mm : bore diameter for boundary conditions
    youngs_modulus_mpa : Young's modulus (default: steel ~200 GPa)
    poissons_ratio : Poisson's ratio (default: 0.3 for steel)
    mesh_spacing : approximate element edge length in mm
    contact_forces : optional list of spiral-impact point forces (dicts with
        'x', 'y', 'fx', 'fy') superimposed on centrifugal body load.
        Produce with ``spiral_contact.contact_forces()``.
    contact_load_mode : "neumann_edge" (default) or "nearest_node".
        Controls how contact forces are distributed to mesh nodes.

    Returns
    -------
    dict with keys:
        peak_stress_mpa : maximum von Mises stress
        mean_stress_mpa : average von Mises stress
        safety_factor : yield_strength / peak_stress
        fea_score : normalized [0, 1] score for optimization
        n_elements : number of mesh elements
        n_nodes : number of mesh nodes
    """
    # Mesh the polygon
    max_area = mesh_spacing ** 2
    nodes, elements = _triangulate_polygon(poly, max_area=max_area)

    if len(elements) < 3:
        # Degenerate mesh — return worst-case
        return {
            "peak_stress_mpa": float("inf"),
            "mean_stress_mpa": float("inf"),
            "safety_factor": 0.0,
            "fea_score": 0.0,
            "n_elements": 0,
            "n_nodes": len(nodes),
        }

    # Material properties
    # Standard FEA mm system: mm, N, MPa, tonne (Mg), s
    # density in tonne/mm³: 7850 kg/m³ = 7.85e-9 tonne/mm³
    E = youngs_modulus_mpa  # N/mm² = MPa
    nu = poissons_ratio
    density_tonne_mm3 = density_kg_m3 * 1e-12  # kg/m³ → tonne/mm³
    omega = rpm * 2 * np.pi / 60.0

    # Assemble stiffness matrix
    K = _assemble_global(nodes, elements, E, nu, thickness_mm)

    # Centrifugal load vector
    F = _centrifugal_load(nodes, elements, omega, density_tonne_mm3, thickness_mm)

    # Superimpose spiral-impact contact forces (optional)
    if contact_forces:
        F = _apply_contact_forces(F, nodes, contact_forces,
                                  mode=contact_load_mode, elements=elements)

    # Boundary conditions
    bore_radius = bore_diameter_mm / 2.0
    K, F, free_dofs = _apply_boundary_conditions(K, F, nodes, bore_radius)

    # Solve K[free] @ u[free] = F[free]
    n_dof = 2 * len(nodes)
    u = np.zeros(n_dof)

    if len(free_dofs) == 0:
        return {
            "peak_stress_mpa": 0.0,
            "mean_stress_mpa": 0.0,
            "safety_factor": float("inf"),
            "fea_score": 1.0,
            "n_elements": len(elements),
            "n_nodes": len(nodes),
        }

    K_free = K[np.ix_(free_dofs, free_dofs)]
    F_free = F[free_dofs]

    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")   # suppress MatrixRankWarning on singular K
            u_free = spsolve(K_free, F_free)
        if not np.isfinite(u_free).all():
            raise ValueError("singular stiffness matrix")
        u[free_dofs] = u_free
    except Exception:
        return {
            "peak_stress_mpa": float("inf"),
            "mean_stress_mpa": float("inf"),
            "safety_factor": 0.0,
            "fea_score": 0.0,
            "n_elements": len(elements),
            "n_nodes": len(nodes),
        }

    # Compute stresses
    vm_stresses = _compute_von_mises(nodes, elements, u, E, nu)

    peak = float(vm_stresses.max()) if len(vm_stresses) > 0 else 0.0
    mean = float(vm_stresses.mean()) if len(vm_stresses) > 0 else 0.0

    # Safety factor
    sf = yield_strength_mpa / peak if peak > 1e-6 else float("inf")

    # FEA score: maps safety factor to [0, 1]
    # sf >= 2.0 → 1.0 (good), sf = 1.0 → 0.5, sf < 1.0 → rapidly drops
    if sf >= 2.0:
        fea_score = 1.0
    elif sf >= 1.0:
        fea_score = 0.5 + 0.5 * (sf - 1.0)
    else:
        fea_score = max(0.0, 0.5 * sf)

    return {
        "peak_stress_mpa": peak,
        "mean_stress_mpa": mean,
        "safety_factor": sf,
        "fea_score": fea_score,
        "n_elements": len(elements),
        "n_nodes": len(nodes),
    }


def fea_stress_analysis_with_mesh(
    poly: Polygon | MultiPolygon,
    rpm: float,
    density_kg_m3: float,
    thickness_mm: float,
    yield_strength_mpa: float,
    bore_diameter_mm: float,
    youngs_modulus_mpa: float = 200_000.0,
    poissons_ratio: float = 0.3,
    mesh_spacing: float = 5.0,
    contact_forces: list[dict] | None = None,
    contact_load_mode: str = "neumann_edge",
) -> dict:
    """Like fea_stress_analysis() but also returns the mesh and per-element stresses.

    Additional keys in the returned dict:
        nodes       : (N, 2) float array of node positions in mm
        elements    : (M, 3) int array of triangle vertex indices
        vm_stresses : (M,) float array of per-element Von Mises stress in MPa

    Parameters
    ----------
    contact_forces : optional list of point forces from spiral contact analysis.
        Each dict must contain 'x', 'y' (mm) and 'fx', 'fy' (N).
        These are superimposed on the centrifugal body load — use
        ``spiral_contact.contact_forces()`` to produce this list.
    """
    max_area = mesh_spacing ** 2
    nodes, elements = _triangulate_polygon(poly, max_area=max_area)

    if len(elements) < 3:
        return {
            "peak_stress_mpa": float("inf"),
            "mean_stress_mpa": float("inf"),
            "safety_factor": 0.0,
            "fea_score": 0.0,
            "n_elements": 0,
            "n_nodes": len(nodes),
            "nodes": nodes,
            "elements": elements,
            "vm_stresses": np.zeros(0),
        }

    E = youngs_modulus_mpa
    nu = poissons_ratio
    density_tonne_mm3 = density_kg_m3 * 1e-12
    omega = rpm * 2 * np.pi / 60.0

    K = _assemble_global(nodes, elements, E, nu, thickness_mm)
    F = _centrifugal_load(nodes, elements, omega, density_tonne_mm3, thickness_mm)
    if contact_forces:
        F = _apply_contact_forces(F, nodes, contact_forces,
                                  mode=contact_load_mode, elements=elements)
    bore_radius = bore_diameter_mm / 2.0
    K, F, free_dofs = _apply_boundary_conditions(K, F, nodes, bore_radius)

    n_dof = 2 * len(nodes)
    u = np.zeros(n_dof)
    vm_stresses = np.zeros(len(elements))

    if len(free_dofs) > 0:
        K_free = K[np.ix_(free_dofs, free_dofs)]
        F_free = F[free_dofs]
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")   # suppress MatrixRankWarning on singular K
                u_free = spsolve(K_free, F_free)
            if np.isfinite(u_free).all():
                u[free_dofs] = u_free
                vm_stresses = _compute_von_mises(nodes, elements, u, E, nu)
        except Exception:
            pass

    peak = float(vm_stresses.max()) if len(vm_stresses) > 0 else 0.0
    mean = float(vm_stresses.mean()) if len(vm_stresses) > 0 else 0.0
    sf = yield_strength_mpa / peak if peak > 1e-6 else float("inf")

    if sf >= 2.0:
        fea_score = 1.0
    elif sf >= 1.0:
        fea_score = 0.5 + 0.5 * (sf - 1.0)
    else:
        fea_score = max(0.0, 0.5 * sf)

    return {
        "peak_stress_mpa": peak,
        "mean_stress_mpa": mean,
        "safety_factor": sf,
        "fea_score": fea_score,
        "n_elements": len(elements),
        "n_nodes": len(nodes),
        "nodes": nodes,
        "elements": elements,
        "vm_stresses": vm_stresses,
    }


def fea_stress_envelope(
    poly: Polygon | MultiPolygon,
    cfg,
    n_patches: int = 12,
) -> dict:
    """Apply n_patches inward radial unit forces simultaneously (single FEA solve).

    Forces are superimposed into one load vector via Neumann-edge distribution,
    then a single solve yields the combined stress field.  KS aggregation (ρ=20)
    produces a smooth envelope stress.

    **Post-analysis only** — not intended for use inside the optimizer loop.

    Parameters
    ----------
    poly      : weapon polygon (mm units)
    cfg       : WeaponConfig — provides mesh spacing and mounting bore
    n_patches : number of equally-spaced inward radial force patches

    Returns
    -------
    dict with keys:
        peak_stress_mpa : maximum von Mises stress under envelope loading
        ks_stress_mpa   : KS-aggregated stress (ρ=20, smooth envelope)
        n_patches       : number of patches applied
    """
    spacing = getattr(cfg.optimization, "fea_coarse_spacing_mm", 10.0)
    max_area = spacing ** 2
    nodes, elements = _triangulate_polygon(poly, max_area=max_area)

    if len(elements) < 3:
        return {"peak_stress_mpa": 0.0, "ks_stress_mpa": 0.0, "n_patches": n_patches}

    E = 200_000.0   # standard steel Young's modulus (MPa)
    nu = 0.3
    thickness = cfg.sheet_thickness_mm

    K = _assemble_global(nodes, elements, E, nu, thickness)
    F = np.zeros(2 * len(nodes))

    # Superimpose n_patches unit inward radial forces via Neumann-edge loading
    if isinstance(poly, MultiPolygon):
        cx = float(poly.centroid.x)
        cy = float(poly.centroid.y)
    else:
        cx = float(poly.centroid.x)
        cy = float(poly.centroid.y)

    max_r = float(np.hypot(nodes[:, 0] - cx, nodes[:, 1] - cy).max())
    angles = np.linspace(0.0, 2.0 * np.pi, n_patches, endpoint=False)

    for angle in angles:
        # Place force point at max_r * 0.95 toward the boundary
        px = cx + max_r * 0.95 * np.cos(angle)
        py = cy + max_r * 0.95 * np.sin(angle)
        # Inward radial direction (unit force)
        fx = -float(np.cos(angle))
        fy = -float(np.sin(angle))
        patch = [{"x": px, "y": py, "fx": fx, "fy": fy}]
        F = _apply_contact_forces(F, nodes, patch,
                                  mode="neumann_edge", elements=elements)

    bore_radius = cfg.mounting.bore_diameter_mm / 2.0
    K, F, free_dofs = _apply_boundary_conditions(K, F, nodes, bore_radius)

    n_dof = 2 * len(nodes)
    u = np.zeros(n_dof)
    if len(free_dofs) > 0:
        K_free = K[np.ix_(free_dofs, free_dofs)]
        F_free = F[free_dofs]
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                u_free = spsolve(K_free, F_free)
            if np.isfinite(u_free).all():
                u[free_dofs] = u_free
        except Exception:
            pass

    vm = _compute_von_mises(nodes, elements, u, E, nu)
    peak = float(vm.max()) if len(vm) > 0 else 0.0

    # KS aggregation with ρ=20 (matches objectives_smooth.py)
    rho = 20.0
    if len(vm) > 0 and vm.max() > 1e-6:
        vm_max = vm.max()
        ks_val = float(vm_max + (1.0 / rho) * np.log(
            np.sum(np.exp(np.clip(rho * (vm - vm_max), -500, 0)))
        ))
    else:
        ks_val = 0.0

    return {"peak_stress_mpa": peak, "ks_stress_mpa": ks_val, "n_patches": n_patches}
