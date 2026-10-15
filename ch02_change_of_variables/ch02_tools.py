"""Chapter 2: changing independent variables, moving frames, similarity variables.

Reusable functions for the Chapter 2 notebook and tests.
"""
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.special import erf


def frozen_pattern_rate(c, dfdx):
    """Local rate of change of a pattern that migrates without changing shape.

    If f(x, t) = F(x - c t), then df/dt = -c * df/dx (Section 2.2 of the study guide).

    Parameters
    ----------
    c : float or array_like
        Migration speed of the pattern, e.g. dune celerity [length/time].
    dfdx : float or array_like
        Spatial gradient of the pattern at the observation point [f/length].

    Returns
    -------
    float or ndarray
        df/dt seen by an observer at a fixed point [f/time].
    """
    return -c * dfdx


def diffusion_length(kappa, t):
    """Diffusion length sqrt(4 kappa t).

    Units: kappa in length^2/time and t in the same time unit; the result is a length.
    """
    return np.sqrt(4.0 * kappa * t)


def erf_step(x, kappa_t, a):
    """Similarity solution for a diffusing step, f = a * erf(x / sqrt(4 kappa t)).

    Solves df/dt = kappa d2f/dx2 on an infinite domain with f(x, 0) = a * sign(x)
    (Worked example in Section 2.3 of the study guide).

    Parameters
    ----------
    x : float or array_like
        Position, measured from the step [length].
    kappa_t : float
        Product kappa * t [length^2]; only this product matters.
    a : float
        Half of the jump across the step.

    Returns
    -------
    float or ndarray
        f(x, t).
    """
    return a * erf(x / np.sqrt(4.0 * kappa_t))


def scarp_profile(x, kappa_t, a, b):
    """Elevation of a diffusively degrading scarp, z = a erf(x / sqrt(4 kappa t)) + b x.

    Parameters
    ----------
    x : float or array_like
        Horizontal distance from the scarp midpoint [m].
    kappa_t : float
        Morphologic age kappa * t [m^2].
    a : float
        Half of the scarp offset [m]; the total offset is 2a.
    b : float
        Far-field (regional) slope [-].

    Returns
    -------
    float or ndarray
        Elevation relative to the scarp midpoint [m].
    """
    return erf_step(x, kappa_t, a) + b * x


def scarp_max_slope(kappa_t, a, b):
    """Steepest slope of the scarp, at x = 0: S_max = a / sqrt(pi kappa t) + b."""
    return a / np.sqrt(np.pi * kappa_t) + b


def scarp_morphologic_age(a, b, s_max):
    """Morphologic age from the steepest slope, kappa t = a^2 / (pi (S_max - b)^2).

    Units: a in m, b and s_max dimensionless; the result is in m^2.
    Divide by a diffusivity (e.g. 1 m^2/kyr) to get an age.
    """
    return a**2 / (np.pi * (s_max - b) ** 2)


def dune_bedload_flux(H, c, porosity, beta=0.5):
    """Mean bedload flux per unit width from dune tracking, q = beta (1 - porosity) c H.

    Assumes dunes that migrate without changing shape and no sediment bypassing the troughs.

    Parameters
    ----------
    H : float
        Dune height [m].
    c : float
        Dune celerity [m/s].
    porosity : float
        Bed porosity lambda_p [-].
    beta : float, optional
        Shape factor; 1/2 for triangular dunes.

    Returns
    -------
    float
        Volume of grains per unit width per unit time [m^2/s].
    """
    return beta * (1.0 - porosity) * c * H


def chi_transform(x, A, A0=1.0, m_over_n=0.5):
    """Chi coordinate of a river profile, chi(x) = integral from 0 to x of (A0 / A)^(m/n) dx'.

    Parameters
    ----------
    x : array_like
        Distance upstream from base level, increasing, with x[0] = 0 at the outlet [m].
    A : array_like
        Drainage area at each x [m^2].
    A0 : float, optional
        Reference drainage area [m^2].
    m_over_n : float, optional
        Concavity index m/n.

    Returns
    -------
    ndarray
        chi at each x, from the trapezoid rule [m].
    """
    integrand = (A0 / np.asarray(A, dtype=float)) ** m_over_n
    return cumulative_trapezoid(integrand, x, initial=0.0)
