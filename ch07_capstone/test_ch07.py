import numpy as np
from scipy.integrate import trapezoid

from ch07_tools import (ashton_flux, ashton_flux_derivative, blowup_time, cerc_flux,
                        cerc_flux_derivative, critical_angle, efolding_time, gaussian_bump,
                        growth_rate, run_one_line, shoreline_diffusivity)


def test_critical_angles():
    assert np.isclose(np.degrees(critical_angle(cerc_flux_derivative)), 45.0)
    a_c = critical_angle(ashton_flux_derivative)                  # Exercise 7.3
    assert np.isclose(np.degrees(a_c), 42.39, atol=0.01)
    assert np.isclose(a_c, np.arctan(np.sqrt(5 / 6)))


def test_flux_derivatives_match_finite_differences():
    alpha, h = np.radians(np.array([10.0, 30.0, 50.0, 70.0])), 1e-6
    for Q, dQ in [(cerc_flux, cerc_flux_derivative), (ashton_flux, ashton_flux_derivative)]:
        numeric = (Q(alpha + h) - Q(alpha - h)) / (2 * h)
        assert np.allclose(numeric, dQ(alpha), rtol=1e-6)
    assert np.allclose(ashton_flux(-alpha), -ashton_flux(alpha))


def test_growth_rates_and_efolding_times():
    # Exercise 7.2(b): |D0| = 0.05 km^2/yr, wavelengths 5 km and 1 km
    D0 = -0.05
    assert np.isclose(efolding_time(5.0, D0), 12.665, rtol=1e-3)
    assert np.isclose(efolding_time(1.0, D0), 0.5066, rtol=1e-3)
    assert growth_rate(2.0, 0.03) < 0 < growth_rate(2.0, -0.03)
    assert np.isclose(shoreline_diffusivity(4e5 * np.cos(np.radians(120)), 10.0), -2e4)


def test_exact_gaussian_runs_backwards_and_conserves_volume():
    x = np.linspace(-3, 3, 6001)
    s0, D0, V = 0.3, -0.02, 1.0
    t_star = blowup_time(s0, D0)
    assert np.isclose(t_star, 2.25)
    for t in (0.0, 0.5 * t_star, 0.9 * t_star):
        assert np.isclose(trapezoid(gaussian_bump(x, t, V, s0, D0), x), V, rtol=1e-6)
    # solves dy/dt = D0 d2y/dx2 (finite-difference check before blow-up)
    t, dt, dx, xs = 1.0, 1e-6, 1e-4, np.array([-0.2, 0.0, 0.25])
    yt = (gaussian_bump(xs, t + dt, V, s0, D0) - gaussian_bump(xs, t - dt, V, s0, D0)) / (2 * dt)
    yxx = (gaussian_bump(xs + dx, t, V, s0, D0) - 2 * gaussian_bump(xs, t, V, s0, D0)
           + gaussian_bump(xs - dx, t, V, s0, D0)) / dx**2
    assert np.allclose(yt, D0 * yxx, rtol=1e-4)


def _sinusoid_run(alpha0_deg, sigma_t, cells_per_wavelength):
    D_sf, K, wavelength, eps = 10.0, 4e5, 2000.0, 0.01          # m, m^3/yr, m, m
    flux = lambda a: cerc_flux(a, K)
    D0 = shoreline_diffusivity(cerc_flux_derivative(np.radians(alpha0_deg), K), D_sf)
    k = 2 * np.pi / wavelength
    dx = wavelength / cells_per_wavelength
    x = np.arange(cells_per_wavelength) * dx
    t_end = sigma_t / abs(D0 * k**2)
    nsteps = 2000
    y = run_one_line(eps * np.sin(k * x), dx, t_end / nsteps, nsteps, np.radians(alpha0_deg),
                     D_sf, flux)
    # amplitude of the sin(kx) mode; discrete second difference replaces k^2
    amp = 2 * np.mean(y * np.sin(k * x))
    k2 = (2 / dx * np.sin(k * dx / 2)) ** 2
    return amp / eps, np.exp(-D0 * k2 * t_end), y


def test_low_angle_waves_smooth_the_coast_at_the_linear_rate():
    ratio, theory, y = _sinusoid_run(20.0, sigma_t=1.0, cells_per_wavelength=64)
    assert ratio < 1 and np.isclose(ratio, theory, rtol=1e-3)


def test_high_angle_waves_roughen_the_coast_at_the_linear_rate():
    ratio, theory, y = _sinusoid_run(60.0, sigma_t=0.3, cells_per_wavelength=16)
    assert ratio > 1 and np.isclose(ratio, theory, rtol=1e-3)


def test_one_line_model_conserves_volume():
    rng = np.random.default_rng(3)
    y0 = 20 * np.exp(-0.5 * ((np.arange(200) * 50.0 - 5000) / 300) ** 2) + rng.normal(0, 0.05, 200)
    y = run_one_line(y0, 50.0, 0.002, 100, np.radians(60), 10.0, lambda a: cerc_flux(a, 4e5))
    assert np.isclose(y.sum(), y0.sum(), rtol=1e-12)
