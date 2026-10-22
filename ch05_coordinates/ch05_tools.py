"""Chapter 5: Cartesian, cylindrical and spherical coordinates; scale factors; a round Earth.

Reusable functions for the Chapter 5 notebook and tests.
"""
import numpy as np
import sympy as sp
from scipy.integrate import trapezoid

EARTH_RADIUS_KM = 6371.0   # mean Earth radius [km]
OMEGA_EARTH = 7.2921e-5    # Earth's rotation rate [rad/s]


# ----------------------------------------------------------------------------
# Scale factors and unit vectors (symbolic)
# ----------------------------------------------------------------------------
def scale_factors_and_unit_vectors(position, coords):
    """Scale factors h_i = |dx/dq_i| and unit vectors e_i = (1/h_i) dx/dq_i.

    Parameters
    ----------
    position : sympy.Matrix
        Cartesian components (x, y, z) written in terms of the new coordinates.
    coords : sequence of three sympy symbols
        The new coordinates (q1, q2, q3). Declare them real or positive.

    Returns
    -------
    h : list of sympy expressions
        Scale factors. Absolute values are dropped, i.e. the scale factors are taken
        positive in the usual coordinate ranges (e.g. R > 0, 0 < theta < pi).
    e : list of sympy.Matrix
        Unit vectors, as Cartesian components.
    """
    h, e = [], []
    for q in coords:
        tangent = position.diff(q)
        hq = sp.sqrt(sp.trigsimp(sp.simplify(tangent.dot(tangent))))
        hq = sp.simplify(hq).replace(sp.Abs, lambda arg: arg)
        h.append(hq)
        e.append(sp.simplify(tangent / hq))
    return h, e


def derivative_in_basis(e, coords, i, j):
    """Components of d(e_i)/d(q_j) along (e_1, e_2, e_3), for an orthonormal basis e."""
    d = e[i].diff(coords[j])
    return [sp.simplify(d.dot(ek)) for ek in e]


# ----------------------------------------------------------------------------
# Volumes of radial landforms
# ----------------------------------------------------------------------------
def radial_volume(thickness, R, angle=2 * np.pi):
    """Volume of a deposit with thickness T(R) filling a sector of the given angle.

    V = angle * integral of T(R) R dR (trapezoid rule). angle = 2 pi gives an
    axisymmetric landform (cone, dome, crater fill); a smaller angle gives a fan.
    Units: T and R in m give V in m^3.
    """
    return angle * trapezoid(np.asarray(thickness, dtype=float) * R, R)


# ----------------------------------------------------------------------------
# A round Earth
# ----------------------------------------------------------------------------
def latlon_cell_area(lat1, lat2, dlon, radius=EARTH_RADIUS_KM):
    """Area of a latitude-longitude cell, R^2 * dlon * (sin lat2 - sin lat1).

    Angles in degrees; the result has the units of radius squared (km^2 by default).
    """
    s1, s2 = np.sin(np.radians(lat1)), np.sin(np.radians(lat2))
    return radius**2 * np.radians(dlon) * (s2 - s1)


def area_weighted_mean(values, lat, dlat):
    """Area-weighted mean of a field on latitude bands of width dlat centred at lat (degrees).

    values has shape (nlat,) or (nlat, nlon) on a regular longitude grid.
    """
    values = np.asarray(values, dtype=float)
    w = latlon_cell_area(np.asarray(lat) - dlat / 2, np.asarray(lat) + dlat / 2, 1.0)
    if values.ndim == 2:
        w = np.broadcast_to(w[:, None], values.shape)
    return np.sum(w * values) / np.sum(w)


def coriolis_parameter(lat, omega=OMEGA_EARTH):
    """Coriolis parameter f = 2 Omega sin(latitude) [1/s]; latitude in degrees."""
    return 2.0 * omega * np.sin(np.radians(lat))


def great_circle_distance(lat1, lon1, lat2, lon2, radius=EARTH_RADIUS_KM):
    """Shortest distance between two points on a sphere (haversine formula).

    Angles in degrees; the result has the units of radius (km by default).
    """
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2.0 * radius * np.arcsin(np.sqrt(a))
