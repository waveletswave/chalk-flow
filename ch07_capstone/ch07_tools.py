"""Chapter 7 (capstone): when diffusion runs backwards -- the high-angle-wave shoreline instability.

Reusable functions for the Chapter 7 notebook and tests. Angles are in radians unless a
name says otherwise.
"""
import numpy as np
from scipy.optimize import brentq


# ----------------------------------------------------------------------------
# Alongshore sediment flux laws Q_s(alpha); alpha = angle between wave crests and shoreline
# ----------------------------------------------------------------------------
def cerc_flux(alpha, K=1.0):
    """Classical angular dependence Q_s = K sin(alpha) cos(alpha) = (K/2) sin(2 alpha)."""
    return K * np.sin(alpha) * np.cos(alpha)


def cerc_flux_derivative(alpha, K=1.0):
    """dQ_s/dalpha = K cos(2 alpha); changes sign at 45 degrees."""
    return K * np.cos(2.0 * alpha)


def ashton_flux(alpha, K=1.0):
    """Refraction-corrected law Q_s = K sin(alpha) cos(alpha)^(6/5) (Ashton and Murray 2006).

    Valid for |alpha| < 90 degrees; odd in alpha.
    """
    alpha = np.asarray(alpha, dtype=float)
    return K * np.sin(alpha) * np.abs(np.cos(alpha)) ** 1.2


def ashton_flux_derivative(alpha, K=1.0):
    """dQ_s/dalpha = K cos(alpha)^(1/5) (cos^2 alpha - (6/5) sin^2 alpha); zero at about 42.4 degrees."""
    alpha = np.asarray(alpha, dtype=float)
    c = np.abs(np.cos(alpha))
    return K * c**0.2 * (np.cos(alpha) ** 2 - 1.2 * np.sin(alpha) ** 2)


def critical_angle(flux_derivative):
    """Angle in (0, 90 deg) where dQ_s/dalpha = 0, i.e. where the shoreline diffusivity changes sign."""
    return brentq(flux_derivative, 1e-6, np.pi / 2 - 1e-6)


# ----------------------------------------------------------------------------
# Linear stability of a straight shoreline
# ----------------------------------------------------------------------------
def shoreline_diffusivity(dQ_dalpha, D_sf, slope=0.0):
    """Effective diffusivity D = (dQ_s/dalpha) / (D_sf (1 + slope^2)) of dy_s/dt = D d2y_s/dx2.

    Units: Q_s in m^3/yr and D_sf in m give D in m^2/yr. Negative D means anti-diffusion.
    """
    return dQ_dalpha / (D_sf * (1.0 + slope**2))


def growth_rate(k, D0):
    """Growth rate sigma = -D0 k^2 of a normal mode exp(i k x + sigma t)."""
    return -D0 * np.asarray(k, dtype=float) ** 2


def efolding_time(wavelength, D0):
    """e-folding time 1 / sigma of a growing mode (D0 < 0), lambda^2 / (4 pi^2 |D0|)."""
    k = 2.0 * np.pi / np.asarray(wavelength, dtype=float)
    return 1.0 / growth_rate(k, D0)


def gaussian_bump(x, t, volume, s0, D0):
    """Exact solution of dy/dt = D0 d2y/dx2 for a Gaussian bump of given volume (plan area).

    The variance is s0^2 + 2 D0 t. For D0 > 0 the bump spreads (Chapter 4's heat kernel);
    for D0 < 0 it sharpens and its height grows without bound as t -> s0^2 / (2 |D0|).
    """
    s2 = s0**2 + 2.0 * D0 * t
    if np.any(np.asarray(s2) <= 0):
        raise ValueError("t is at or beyond the blow-up time s0^2 / (2 |D0|)")
    x = np.asarray(x, dtype=float)
    return volume * np.exp(-0.5 * x**2 / s2) / np.sqrt(2.0 * np.pi * s2)


def blowup_time(s0, D0):
    """Time at which a Gaussian bump under anti-diffusion (D0 < 0) becomes a spike."""
    if D0 >= 0:
        return np.inf
    return s0**2 / (2.0 * abs(D0))


# ----------------------------------------------------------------------------
# A one-line shoreline model (periodic coast, explicit finite volumes)
# ----------------------------------------------------------------------------
def shoreline_angle(y, dx):
    """Shoreline orientation theta_s = arctan(dy/dx) at the faces between cells i and i+1."""
    return np.arctan((np.roll(y, -1) - y) / dx)


def step_one_line(y, dx, dt, alpha0, D_sf, flux):
    """One explicit step of D_sf dy/dt = -dQ_s/dx on a periodic coast.

    flux is a callable Q_s(alpha) evaluated at the cell faces, with alpha = alpha0 - theta_s.
    The flux form conserves the total shoreline volume exactly.
    """
    Q = flux(alpha0 - shoreline_angle(y, dx))
    return y - dt / (D_sf * dx) * (Q - np.roll(Q, 1))


def run_one_line(y0, dx, dt, nsteps, alpha0, D_sf, flux, save_every=None):
    """Run the one-line model for nsteps; optionally return snapshots every save_every steps.

    Returns the final shoreline, or (times, snapshots) if save_every is given.
    For alpha0 below the critical angle, explicit stability needs dt < dx^2 / (2 D).
    """
    y = np.array(y0, dtype=float)
    times, snaps = [0.0], [y.copy()]
    for n in range(1, nsteps + 1):
        y = step_one_line(y, dx, dt, alpha0, D_sf, flux)
        if save_every and n % save_every == 0:
            times.append(n * dt)
            snaps.append(y.copy())
    if save_every:
        return np.array(times), np.array(snaps)
    return y
