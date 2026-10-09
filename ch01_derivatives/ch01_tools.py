"""Chapter 1: partial and total derivatives, material derivative, characteristics.

Reusable functions for the Chapter 1 notebook and tests.
"""
import numpy as np


def material_derivative(dCdt, u, dCdx):
    """One-dimensional material derivative, DC/Dt = dC/dt + u * dC/dx.

    The derivatives on the right-hand side are partial derivatives.

    Parameters
    ----------
    dCdt : float or array
        Local (Eulerian) rate of change at a fixed point.
    u : float or array
        Flow velocity.
    dCdx : float or array
        Spatial gradient in the flow direction.

    Returns
    -------
    float or array
        Rate of change following a fluid parcel (Lagrangian).
    """
    return dCdt + u * dCdx


def knickpoint_travel_time(K, ka, L, ell):
    """Time for a knickpoint to travel from the outlet to a distance ell from the divide.

    Assumes stream-power incision with m = 1/2 and n = 1, and a drainage area
    A(x) = ka * (L - x)**2, where x is measured upstream from the outlet
    (Exercise 1.4 of the study guide).

    Units: K in 1/yr, ka dimensionless, L and ell in m; the result is in years.
    """
    return np.log(L / ell) / (K * np.sqrt(ka))
