"""Chapter 6: equations in curvilinear coordinates, exact flows, and their field applications.

Reusable functions for the Chapter 6 notebook and tests.

The symbolic operators work in any orthogonal, right-handed coordinate system (q1, q2, q3)
with scale factors h = (h1, h2, h3), e.g. (1, R, 1) for cylindrical (R, phi, z) and
(1, r, r sin(theta)) for spherical (r, theta, phi) coordinates. Vector fields are lists of
their three physical components (u1, u2, u3) along the local unit vectors.
"""
import numpy as np
import sympy as sp
from scipy.special import exp1


# ----------------------------------------------------------------------------
# Master formulas (symbolic): every operator from the scale factors
# ----------------------------------------------------------------------------
def grad(f, coords, h):
    """Gradient of a scalar: components (1/h_i) df/dq_i."""
    return [sp.diff(f, q) / hi for q, hi in zip(coords, h)]


def div(u, coords, h):
    """Divergence: (1/H) sum_i d/dq_i (H u_i / h_i), with H = h1 h2 h3."""
    H = h[0] * h[1] * h[2]
    return sum(sp.diff(H / h[i] * u[i], coords[i]) for i in range(3)) / H


def laplacian(f, coords, h):
    """Laplacian of a scalar: (1/H) sum_i d/dq_i (H / h_i^2 df/dq_i)."""
    H = h[0] * h[1] * h[2]
    return sum(sp.diff(H / h[i] ** 2 * sp.diff(f, coords[i]), coords[i]) for i in range(3)) / H


def curl(u, coords, h):
    """Curl of a vector field (physical components), from the determinant formula."""
    q = coords
    return [
        (sp.diff(h[2] * u[2], q[1]) - sp.diff(h[1] * u[1], q[2])) / (h[1] * h[2]),
        (sp.diff(h[0] * u[0], q[2]) - sp.diff(h[2] * u[2], q[0])) / (h[2] * h[0]),
        (sp.diff(h[1] * u[1], q[0]) - sp.diff(h[0] * u[0], q[1])) / (h[0] * h[1]),
    ]


def vector_laplacian(u, coords, h):
    """Vector Laplacian, defined safely as grad(div u) - curl(curl u)."""
    g = grad(div(u, coords, h), coords, h)
    cc = curl(curl(u, coords, h), coords, h)
    return [g[i] - cc[i] for i in range(3)]


def advect_scalar(f, u, coords, h):
    """The scalar operator (u . grad) acting on a function f: sum_i (u_i / h_i) df/dq_i."""
    return sum(u[i] / h[i] * sp.diff(f, coords[i]) for i in range(3))


def convective_acceleration(u, coords, h):
    """Components of the vector (u . grad) u, from the Lamb form grad(|u|^2/2) - u x curl(u).

    Unlike advect_scalar applied to each component, this includes the terms that come
    from turning unit vectors (centripetal, angular momentum, cot(theta) terms).
    """
    w = curl(u, coords, h)
    g = grad((u[0] ** 2 + u[1] ** 2 + u[2] ** 2) / 2, coords, h)
    cross = [u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0]]
    return [g[i] - cross[i] for i in range(3)]


# ----------------------------------------------------------------------------
# Exact flows
# ----------------------------------------------------------------------------
def rankine_vortex(R, a, U_m):
    """Azimuthal velocity of a Rankine vortex: rigid core (R <= a), potential vortex outside.

    U_m is the maximum speed, reached at R = a. Units follow the inputs.
    """
    R = np.asarray(R, dtype=float)
    return np.where(R <= a, U_m * R / a, U_m * a / np.maximum(R, a))


def rankine_pressure(R, a, U_m, rho):
    """Pressure relative to far field, p - p_inf, in a Rankine vortex (from dp/dR = rho u^2 / R).

    The central deficit is rho U_m^2. Units: rho in kg/m^3 and U_m in m/s give Pa.
    """
    R = np.asarray(R, dtype=float)
    inside = rho * U_m**2 * (R**2 / (2 * a**2) - 1.0)
    outside = -rho * U_m**2 * a**2 / (2 * np.maximum(R, a) ** 2)
    return np.where(R <= a, inside, outside)


def poiseuille_velocity(R, a, G, mu):
    """Hagen-Poiseuille velocity u_z = G (a^2 - R^2) / (4 mu); G = -dp/dz [Pa/m], mu [Pa s]."""
    return G * (a**2 - np.asarray(R, dtype=float) ** 2) / (4.0 * mu)


def poiseuille_discharge(a, G, mu):
    """Discharge through a pipe of radius a, Q = pi G a^4 / (8 mu) [m^3/s]."""
    return np.pi * G * a**4 / (8.0 * mu)


def stokes_flow_sphere(r, theta, a, U):
    """Velocity (u_r, u_theta) of creeping flow U e_z past a sphere of radius a (Stokes 1851)."""
    r = np.asarray(r, dtype=float)
    ur = U * np.cos(theta) * (1 - 1.5 * a / r + 0.5 * a**3 / r**3)
    ut = -U * np.sin(theta) * (1 - 0.75 * a / r - 0.25 * a**3 / r**3)
    return ur, ut


def stokes_stream_function(r, theta, a, U):
    """Stokes stream function psi = (U/2) r^2 sin^2(theta) (1 - 3a/(2r) + a^3/(2r^3)).

    u_r = (1 / (r^2 sin theta)) dpsi/dtheta and u_theta = -(1 / (r sin theta)) dpsi/dr.
    """
    r = np.asarray(r, dtype=float)
    return 0.5 * U * r**2 * np.sin(theta) ** 2 * (1 - 1.5 * a / r + 0.5 * a**3 / r**3)


# ----------------------------------------------------------------------------
# Grains, bends, wells and domes
# ----------------------------------------------------------------------------
def stokes_settling_velocity(d, Rs=1.65, g=9.81, nu=1.0e-6):
    """Stokes' law w_s = Rs g d^2 / (18 nu) [m/s]; valid for Re_p = w_s d / nu below about 1."""
    return Rs * g * np.asarray(d, dtype=float) ** 2 / (18.0 * nu)


def ferguson_church_settling_velocity(d, Rs=1.65, g=9.81, nu=1.0e-6, C1=18.0, C2=1.0):
    """Settling velocity of natural grains (Ferguson and Church 2004) [m/s].

    w_s = Rs g d^2 / (C1 nu + sqrt(0.75 C2 Rs g d^3)); C1 = 18, C2 = 1.0 for natural sand
    with sieve diameters (C2 = 0.4 for smooth spheres). Tends to Stokes' law for fine grains.
    """
    d = np.asarray(d, dtype=float)
    return Rs * g * d**2 / (C1 * nu + np.sqrt(0.75 * C2 * Rs * g * d**3))


def rouse_number(ws, u_star, kappa=0.41):
    """Rouse number P = w_s / (kappa u_*)."""
    return ws / (kappa * u_star)


def rouse_profile(z, h, a, P):
    """Suspended-sediment concentration relative to the reference level a, c / c_a.

    c / c_a = [((h - z) / z) (a / (h - a))]^P for a <= z < h (Rouse 1937).
    """
    z = np.asarray(z, dtype=float)
    return (((h - z) / z) * (a / (h - a))) ** P


def superelevation(U, B, Rc, g=9.81):
    """Cross-channel rise of the water surface in a bend, U^2 B / (g Rc) [m]."""
    return U**2 * B / (g * Rc)


def thiem_head(r, rw, hw, Q, T):
    """Steady head around a well in a confined aquifer, h = h_w + Q / (2 pi T) ln(r / r_w)."""
    return hw + Q / (2.0 * np.pi * T) * np.log(np.asarray(r, dtype=float) / rw)


def theis_drawdown(r, t, Q, T, S):
    """Theis drawdown s = Q / (4 pi T) E1(r^2 S / (4 T t)).

    Units: Q in m^3/s, T in m^2/s, S dimensionless, r in m and t in s give s in m.
    """
    u = np.asarray(r, dtype=float) ** 2 * S / (4.0 * T * np.asarray(t, dtype=float))
    return Q / (4.0 * np.pi * T) * exp1(u)


def steady_dome_profile(R, U, kappa):
    """Steady diffusive dome, elevation below the summit: -U R^2 / (4 kappa).

    Solves 0 = U + kappa (1/R) d/dR (R dz/dR). Its relief is half that of a ridge of the
    same half-width, whose profile is -U x^2 / (2 kappa).
    """
    return -U * np.asarray(R, dtype=float) ** 2 / (4.0 * kappa)
