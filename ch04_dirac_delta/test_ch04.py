import numpy as np
from scipy.integrate import trapezoid
from scipy.special import erf

from ch04_tools import (coarse_grain, diffuse_by_convolution, gaussian_delta, heat_kernel,
                        nash_cascade_iuh, profile_moments, rainfall_runoff, sifting_integral,
                        superpose_impulses, tracer_concentration, velocity_and_dispersion)


def test_gaussian_delta_has_unit_area_and_sifts():
    x = np.linspace(-20, 20, 400001)
    f = lambda s: np.cos(s) + s**2
    for sigma in [1.0, 0.3, 0.1]:
        assert np.isclose(trapezoid(gaussian_delta(x, sigma, 1.3), x), 1.0)
        exact = np.exp(-sigma**2 / 2) * np.cos(1.3) + 1.3**2 + sigma**2
        assert np.isclose(sifting_integral(f, x, 1.3, sigma), exact, rtol=1e-8)


def test_scaling_property():
    # integral of e^x delta(2x + 4) dx = e^(-2) / 2  (Exercise 4.1(d))
    x = np.linspace(-6, 2, 800001)
    value = trapezoid(np.exp(x) * gaussian_delta(2 * x + 4, 1e-3), x)
    assert np.isclose(value, 0.5 * np.exp(-2), rtol=1e-5)


def test_heat_kernel_solves_diffusion_with_variance_2kt():
    kappa, t = 0.8, 2.0
    x = np.linspace(-30, 30, 60001)
    G = heat_kernel(x, kappa * t)
    m0, mean, var = profile_moments(x, G)
    assert np.isclose(m0, 1.0) and np.isclose(mean, 0.0, atol=1e-12)
    assert np.isclose(var, 2 * kappa * t)
    dt, dx = 1e-5, 1e-3
    xs = np.array([-2.0, 0.0, 1.5])
    Gt = (heat_kernel(xs, kappa * (t + dt)) - heat_kernel(xs, kappa * (t - dt))) / (2 * dt)
    Gxx = (heat_kernel(xs + dx, kappa * t) - 2 * heat_kernel(xs, kappa * t)
           + heat_kernel(xs - dx, kappa * t)) / dx**2
    assert np.allclose(Gt, kappa * Gxx, rtol=1e-4)


def test_convolving_a_step_gives_erf():
    # Exercise 4.3(b): c0 = a sign(x) -> a erf(x / sqrt(4 kappa t))
    x = np.linspace(-40, 40, 2001)
    a, kt = 2.0, 3.0
    c = diffuse_by_convolution(a * np.sign(x), x, kt)
    inner = np.abs(x) < 20
    assert np.allclose(c[inner], a * erf(x[inner] / np.sqrt(4 * kt)), atol=1e-4)


def test_tracer_method_of_moments():
    # Exercise 4.4(a): centroids 720 m and 2160 m, variances 400 and 1600 m^2 at 10 and 30 min
    U, Kx = velocity_and_dispersion(600, 720, 400, 1800, 2160, 1600)
    assert np.isclose(U, 1.2) and np.isclose(Kx, 0.5)
    # synthetic profiles with a virtual origin are recovered exactly
    x = np.linspace(0, 4000, 40001)
    moments = [profile_moments(x, tracer_concentration(x, t, 5.0, 20.0, 1.2, 0.5, t0=200.0))
               for t in (600.0, 1800.0)]
    (m0a, ma, va), (m0b, mb, vb) = moments
    assert np.isclose(m0a, 0.25) and np.isclose(ma, 720) and np.isclose(va, 400)
    assert np.allclose(velocity_and_dispersion(600, ma, va, 1800, mb, vb), (1.2, 0.5))


def test_linear_reservoir_two_storms():
    # Exercise 4.5(b): k = 5 h, P1 = 20 mm at t = 0, P2 = 10 mm at t = 3 h
    response = lambda s: nash_cascade_iuh(s, k=5.0, n=1)
    Q = superpose_impulses(np.array([0.0, 3.0]), [0.0, 3.0], [20.0, 10.0], response)
    assert np.isclose(Q[0], 4.0)
    assert np.isclose(Q[1], (20 * np.exp(-0.6) + 10) / 5)


def test_nash_cascade_area_and_peak():
    t = np.linspace(0, 200, 200001)
    for n, k in [(1, 2.0), (3, 1.5), (5, 0.8)]:
        u = nash_cascade_iuh(t, k, n)
        assert np.isclose(trapezoid(u, t), 1.0, rtol=1e-6)
        if n > 1:
            assert np.isclose(t[np.argmax(u)], (n - 1) * k, atol=1e-3)
    assert nash_cascade_iuh(-1.0, 2.0, 3) == 0.0


def test_constant_rain_approaches_steady_runoff():
    dt, k, rain = 0.01, 3.0, 4.0
    t = np.arange(0, 30, dt)
    Q = rainfall_runoff(np.full(t.size, rain), dt, nash_cascade_iuh(t, k, 1))
    assert np.allclose(Q[100:], rain * (1 - np.exp(-t[100:] / k)), rtol=5e-3)


def test_coarse_grain_conserves_particles_and_velocity():
    rng = np.random.default_rng(1)
    positions = rng.uniform(40, 60, 300)
    x = np.linspace(0, 100, 5001)
    n = coarse_grain(x, positions, sigma=2.0)
    assert np.isclose(trapezoid(n, x), 300, rtol=1e-6)
    J = coarse_grain(x, positions, sigma=2.0, weights=np.full(300, 0.3))
    inside = n > 1e-3
    assert np.allclose(J[inside] / n[inside], 0.3)
