import numpy as np

from ch02_tools import (chi_transform, diffusion_length, dune_bedload_flux, erf_step,
                        frozen_pattern_rate, scarp_max_slope, scarp_morphologic_age,
                        scarp_profile)


def test_frozen_pattern_rate_on_dune_faces():
    # dunes migrating at 1.5 m/h: the gentle stoss side erodes, the steep lee side fills
    assert np.isclose(frozen_pattern_rate(1.5, 0.05), -0.075)
    assert np.isclose(frozen_pattern_rate(1.5, -0.6), 0.9)


def test_erf_step_solves_the_diffusion_equation():
    # finite-difference check of df/dt = kappa d2f/dx2 at a few points
    kappa, a, t = 0.7, 2.0, 3.0
    x = np.array([-1.5, -0.3, 0.4, 2.0])
    dt, dx = 1e-5, 1e-3

    def f(xx, tt):
        return erf_step(xx, kappa * tt, a)

    dfdt = (f(x, t + dt) - f(x, t - dt)) / (2 * dt)
    d2fdx2 = (f(x + dx, t) - 2 * f(x, t) + f(x - dx, t)) / dx**2
    assert np.allclose(dfdt, kappa * d2fdx2, rtol=1e-4, atol=1e-8)


def test_similarity_scaling():
    # doubling x and quadrupling t gives the same value: f depends on x / sqrt(t) only
    x = np.linspace(-5, 5, 11)
    assert np.allclose(erf_step(x, 3.0, 1.0), erf_step(2 * x, 12.0, 1.0))
    assert np.isclose(diffusion_length(1.0, 10.0), np.sqrt(40.0))


def test_scarp_dating_example():
    # Exercise 2.4(b): 2a = 4 m, b = 0.05, S_max = 0.30 -> kappa t = 20.37 m^2
    kt = scarp_morphologic_age(a=2.0, b=0.05, s_max=0.30)
    assert np.isclose(kt, 20.372, atol=1e-3)
    assert np.isclose(scarp_max_slope(kt, a=2.0, b=0.05), 0.30)


def test_scarp_max_slope_matches_numerical_gradient():
    kt, a, b, h = 20.0, 2.0, 0.05, 1e-4
    numerical = (scarp_profile(h, kt, a, b) - scarp_profile(-h, kt, a, b)) / (2 * h)
    assert np.isclose(numerical, scarp_max_slope(kt, a, b), rtol=1e-6)


def test_dune_bedload_flux():
    # Exercise 2.5: H = 0.8 m, c = 1.5 m/h, porosity 0.35 -> 1.08e-4 m^2/s
    q = dune_bedload_flux(H=0.8, c=1.5 / 3600, porosity=0.35)
    assert np.isclose(q, 1.0833e-4, rtol=1e-3)


def test_chi_transform_matches_analytic_result():
    # A = ka (L - x)^2 and m/n = 1/2 give chi = sqrt(A0 / ka) * ln(L / (L - x))
    ka, L, A0 = 1.0, 20e3, 1.0
    x = np.linspace(0, 19e3, 20001)
    chi = chi_transform(x, ka * (L - x) ** 2, A0=A0, m_over_n=0.5)
    exact = np.sqrt(A0 / ka) * np.log(L / (L - x))
    assert np.allclose(chi, exact, rtol=1e-6, atol=1e-6)
