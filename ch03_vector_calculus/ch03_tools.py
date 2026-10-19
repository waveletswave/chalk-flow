"""Chapter 3: vector calculus, index notation, integral theorems and landscape fluxes.

Reusable functions for the Chapter 3 notebook and tests.

Grid convention: 2-D arrays have shape (ny, nx) with f[j, i] = f(x[i], y[j]),
as returned by np.meshgrid(x, y). Grid spacings dx and dy are uniform.
"""
import numpy as np
from scipy.integrate import trapezoid


# ----------------------------------------------------------------------------
# Index notation
# ----------------------------------------------------------------------------
def levi_civita():
    """Alternating symbol epsilon_ijk as a 3 x 3 x 3 array.

    +1 for even permutations of (0, 1, 2), -1 for odd ones, 0 if an index repeats.
    """
    eps = np.zeros((3, 3, 3))
    for i, j, k in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]:
        eps[i, j, k] = 1.0
        eps[i, k, j] = -1.0
    return eps


def cross_index(a, b):
    """Cross product written in index notation, (a x b)_i = epsilon_ijk a_j b_k."""
    return np.einsum("ijk,j,k->i", levi_civita(), a, b)


# ----------------------------------------------------------------------------
# Differential operators on a 2-D grid (second-order finite differences)
# ----------------------------------------------------------------------------
def grad_2d(f, dx, dy):
    """Gradient of a gridded scalar field.

    Returns
    -------
    tuple of ndarray
        (df/dx, df/dy), each with the shape of f.
    """
    dfdy, dfdx = np.gradient(f, dy, dx, edge_order=2)
    return dfdx, dfdy


def div_2d(qx, qy, dx, dy):
    """Divergence dqx/dx + dqy/dy of a gridded 2-D vector field."""
    return (np.gradient(qx, dx, axis=1, edge_order=2)
            + np.gradient(qy, dy, axis=0, edge_order=2))


def curl_z_2d(u, v, dx, dy):
    """Vertical component of the curl, dv/dx - du/dy (the vorticity of a plane flow)."""
    return (np.gradient(v, dx, axis=1, edge_order=2)
            - np.gradient(u, dy, axis=0, edge_order=2))


def laplacian_2d(f, dx, dy):
    """Laplacian of a gridded scalar field, computed as div(grad f)."""
    fx, fy = grad_2d(f, dx, dy)
    return div_2d(fx, fy, dx, dy)


# ----------------------------------------------------------------------------
# Integral theorems on a rectangle (trapezoid rule)
# ----------------------------------------------------------------------------
def area_integral(f, x, y):
    """Double integral of a gridded field over the rectangle spanned by x and y."""
    return trapezoid(trapezoid(f, x, axis=1), y)


def boundary_flux_rectangle(qx, qy, x, y):
    """Net outward flux of (qx, qy) through the boundary of the rectangle (Gauss' theorem).

    The outward normal is +x on the right edge, -x on the left, +y on the top and -y
    on the bottom.
    """
    right = trapezoid(qx[:, -1], y)
    left = trapezoid(qx[:, 0], y)
    top = trapezoid(qy[-1, :], x)
    bottom = trapezoid(qy[0, :], x)
    return right - left + top - bottom


def circulation_rectangle(u, v, x, y):
    """Counterclockwise circulation of (u, v) around the rectangle (Stokes' theorem)."""
    bottom = trapezoid(u[0, :], x)     # along +x at y = y[0]
    right = trapezoid(v[:, -1], y)     # along +y at x = x[-1]
    top = -trapezoid(u[-1, :], x)      # along -x at y = y[-1]
    left = -trapezoid(v[:, 0], y)      # along -y at x = x[0]
    return bottom + right + top + left


# ----------------------------------------------------------------------------
# Landscapes as the divergence of fluxes
# ----------------------------------------------------------------------------
def exner_rate(div_qs, porosity, source=0.0):
    """Rate of bed elevation change from the Exner equation.

    (1 - porosity) dz_b/dt = -div(q_s) + source

    Parameters
    ----------
    div_qs : float or array_like
        Divergence of the volumetric sediment flux per unit width [m/s or m/yr].
    porosity : float
        Bed porosity lambda_p [-].
    source : float or array_like, optional
        Volume of grains supplied per unit bed area per unit time [same units as div_qs].

    Returns
    -------
    float or ndarray
        dz_b/dt; positive means the bed rises (aggradation).
    """
    return (-np.asarray(div_qs) + source) / (1.0 - porosity)


def steady_ridge_linear(x, U, kappa):
    """Steady ridge for linear diffusion, elevation below the crest: -U x^2 / (2 kappa).

    Solves 0 = U + kappa d2z/dx2 with the crest at x = 0.
    Units: U in m/yr, kappa in m^2/yr, x in m; the result is in m.
    """
    return -U * np.asarray(x, dtype=float) ** 2 / (2.0 * kappa)


def nonlinear_hillslope_flux(S, kappa, S_c):
    """Downslope soil flux for the nonlinear law q = kappa S / (1 - (S / S_c)^2).

    S is the downslope gradient (|S| < S_c); the flux grows without bound as |S|
    approaches the critical gradient S_c (Roering et al. 1999).
    """
    S = np.asarray(S, dtype=float)
    return kappa * S / (1.0 - (S / S_c) ** 2)


def steady_ridge_nonlinear(x, U, kappa, S_c):
    """Steady ridge for the nonlinear flux law, elevation below the crest.

    Analytic solution of 0 = U - d/dx[q] with q from nonlinear_hillslope_flux and the
    crest at x = 0 (Roering et al. 2007):
    z = -(S_c^2 / (2 beta)) [w - 1 - ln((1 + w) / 2)], w = sqrt(1 + (2 beta x / S_c)^2),
    beta = U / kappa.
    """
    beta = U / kappa
    w = np.sqrt(1.0 + (2.0 * beta * np.asarray(x, dtype=float) / S_c) ** 2)
    return -(S_c ** 2 / (2.0 * beta)) * (w - 1.0 - np.log((1.0 + w) / 2.0))


def erosion_rate_from_hilltop_curvature(C_ht, kappa, density_ratio=1.0):
    """Steady-state erosion rate from hilltop curvature, E = -density_ratio * kappa * C_ht.

    C_ht is the Laplacian of elevation at the hilltop [1/m] (negative on a convex crest).
    Use density_ratio = rho_soil / rho_rock when kappa is a soil diffusivity.
    Units: kappa in m^2/yr gives E in m/yr.
    """
    return -density_ratio * kappa * C_ht


def flow_divergence(z, dx, dy):
    """Divergence of the unit downslope direction, div(-grad z / |grad z|) [1/m].

    Positive where flow lines spread apart (noses and ridges), negative where they
    converge (hollows and valleys). Undefined on flat cells.
    """
    zx, zy = grad_2d(z, dx, dy)
    g = np.hypot(zx, zy)
    return div_2d(-zx / g, -zy / g, dx, dy)
