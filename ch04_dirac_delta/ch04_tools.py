"""Chapter 4: the Dirac delta function, Green's functions and impulse responses.

Reusable functions for the Chapter 4 notebook and tests.
"""
from math import factorial

import numpy as np
from scipy.integrate import trapezoid


# ----------------------------------------------------------------------------
# The delta function as a limit
# ----------------------------------------------------------------------------
def gaussian_delta(x, sigma, x0=0.0):
    """Unit-area Gaussian delta_sigma(x - x0) (Eq. B.4.1); tends to delta(x - x0) as sigma -> 0.

    Units: 1 / [x].
    """
    s = (np.asarray(x, dtype=float) - x0) / sigma
    return np.exp(-0.5 * s**2) / (np.sqrt(2.0 * np.pi) * sigma)


def sifting_integral(f, x, x0, sigma):
    """Numerical value of the integral of f(x) delta_sigma(x - x0) dx on the grid x.

    Tends to f(x0) as sigma -> 0 (sifting property, Eq. B.4.4). f is a callable.
    The grid must resolve sigma (spacing well below sigma).
    """
    x = np.asarray(x, dtype=float)
    return trapezoid(f(x) * gaussian_delta(x, sigma, x0), x)


# ----------------------------------------------------------------------------
# Green's function of the diffusion equation
# ----------------------------------------------------------------------------
def heat_kernel(x, kappa_t):
    """Impulse response of dc/dt = kappa d2c/dx2 on an infinite line.

    G = exp(-x^2 / (4 kappa t)) / sqrt(4 pi kappa t), a Gaussian with variance 2 kappa t.
    Only the product kappa_t = kappa * t [length^2] matters. Units: 1 / [x].
    """
    return gaussian_delta(x, np.sqrt(2.0 * kappa_t))


def diffuse_by_convolution(c0, x, kappa_t):
    """Diffuse an initial profile by convolving it with the heat kernel.

    c(x, t) = integral of G(x - xi, t) c0(xi) dxi, evaluated with trapezoid weights.
    Accurate away from the ends of the grid (more than a few diffusion lengths
    sqrt(4 kappa t) from them). Builds an n x n matrix, so keep n below a few thousand.
    """
    x = np.asarray(x, dtype=float)
    w = np.gradient(x)                 # spacing around each node
    w[0] *= 0.5                         # trapezoid end weights
    w[-1] *= 0.5
    G = heat_kernel(x[:, None] - x[None, :], kappa_t)
    return G @ (np.asarray(c0, dtype=float) * w)


# ----------------------------------------------------------------------------
# Tracer tests
# ----------------------------------------------------------------------------
def tracer_concentration(x, t, M, A, U, Kx, t0=0.0):
    """Concentration after an instantaneous release of mass M across a channel of area A.

    Green's function of 1-D advection-dispersion,
    c = (M / A) / sqrt(2 pi s2) * exp(-(x - U t)^2 / (2 s2)),  s2 = 2 Kx (t - t0).
    t0 is a virtual origin for the spreading. Units: M in kg, A in m^2, U in m/s,
    Kx in m^2/s, x in m and t in s give c in kg/m^3 (= g/L).
    """
    s2 = 2.0 * Kx * (t - t0)
    return (M / A) * gaussian_delta(x, np.sqrt(s2), U * t)


def profile_moments(x, c):
    """Zeroth moment, centroid and variance of a concentration profile c(x)."""
    x = np.asarray(x, dtype=float)
    m0 = trapezoid(c, x)
    mean = trapezoid(x * c, x) / m0
    var = trapezoid((x - mean) ** 2 * c, x) / m0
    return m0, mean, var


def velocity_and_dispersion(t1, mean1, var1, t2, mean2, var2):
    """Mean velocity and longitudinal dispersion coefficient from two profiles.

    U = d(mean)/dt and Kx = (1/2) d(variance)/dt, using differences so that the
    unknown initial spreading cancels (method of moments).
    """
    U = (mean2 - mean1) / (t2 - t1)
    Kx = 0.5 * (var2 - var1) / (t2 - t1)
    return U, Kx


# ----------------------------------------------------------------------------
# Unit hydrographs
# ----------------------------------------------------------------------------
def nash_cascade_iuh(t, k, n=1):
    """Instantaneous unit hydrograph of a cascade of n linear reservoirs (Nash 1957).

    u(t) = t^(n-1) exp(-t/k) / (k^n (n-1)!) for t >= 0, and 0 before. n = 1 is a single
    linear reservoir, u = exp(-t/k) / k. Unit area; peak at t = (n - 1) k; mean lag n k.
    Units: 1 / [t].
    """
    t = np.asarray(t, dtype=float)
    tp = np.clip(t, 0.0, None)
    u = tp ** (n - 1) * np.exp(-tp / k) / (k**n * factorial(n - 1))
    return np.where(t >= 0, u, 0.0)


def superpose_impulses(t, times, amounts, response):
    """Response to a train of impulses, sum_i a_i * response(t - t_i).

    response is a callable that returns 0 for negative arguments (causality).
    """
    t = np.asarray(t, dtype=float)
    return sum(a * response(t - ti) for ti, a in zip(times, amounts))


def rainfall_runoff(p, dt, u):
    """Discharge from rainfall by convolution, Q(t) = integral from 0 to t of u(t - tau) p(tau) dtau.

    p and u are sampled at the same interval dt, starting at t = 0, and u is at least as
    long as p. The integral is evaluated with the trapezoid rule. The result has the
    length of p and the units of p (e.g. mm/h).
    """
    p = np.asarray(p, dtype=float)
    u = np.asarray(u, dtype=float)[: len(p)]
    if len(u) < len(p):
        raise ValueError("u must be at least as long as p")
    full = np.convolve(p, u)[: len(p)]
    return (full - 0.5 * (p[0] * u + p * u[0])) * dt


# ----------------------------------------------------------------------------
# From particles to continuum fields
# ----------------------------------------------------------------------------
def coarse_grain(x, positions, sigma, weights=None):
    """Smoothed field sum_p w_p delta_sigma(x - x_p) from particle positions.

    With weights = None this is the number density (particles per unit length);
    with weights = particle velocities it is the particle flux.
    """
    x = np.asarray(x, dtype=float)
    positions = np.asarray(positions, dtype=float)
    w = np.ones_like(positions) if weights is None else np.asarray(weights, dtype=float)
    return (w[None, :] * gaussian_delta(x[:, None], sigma, positions[None, :])).sum(axis=1)
