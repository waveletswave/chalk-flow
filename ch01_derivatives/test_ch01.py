"""Checks for ch01_tools against values worked out in the study guide."""
import numpy as np
from ch01_tools import material_derivative, knickpoint_travel_time


def test_bridge_vs_leaf():
    # Worked example 1.1: the gauge sees -0.020, a water parcel sees -0.0027
    # (kg m^-3 per hour; u = 4.32 km/h, dC/dx = 0.004 kg m^-3 per km)
    DCDt = material_derivative(dCdt=-0.020, u=4.32, dCdx=0.004)
    assert np.isclose(DCDt, -0.00272)


def test_knickpoint_travel_time():
    # Exercise 1.4: about 1.4979 million years
    t = knickpoint_travel_time(K=2e-6, ka=1.0, L=20e3, ell=1e3)
    assert np.isclose(t / 1e6, 1.4979, atol=1e-4)
