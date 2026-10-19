import numpy as np

from ch03_tools import (area_integral, boundary_flux_rectangle, circulation_rectangle,
                        cross_index, curl_z_2d, div_2d, erosion_rate_from_hilltop_curvature,
                        exner_rate, flow_divergence, grad_2d, laplacian_2d, levi_civita,
                        nonlinear_hillslope_flux, steady_ridge_linear, steady_ridge_nonlinear)


def test_cross_product_in_index_notation():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=3), rng.normal(size=3)
    assert np.allclose(cross_index(a, b), np.cross(a, b))


def test_epsilon_delta_identity():
    # epsilon_ijk epsilon_ilm = delta_jl delta_km - delta_jm delta_kl
    eps, d = levi_civita(), np.eye(3)
    lhs = np.einsum("ijk,ilm->jklm", eps, eps)
    rhs = np.einsum("jl,km->jklm", d, d) - np.einsum("jm,kl->jklm", d, d)
    assert np.allclose(lhs, rhs)


def test_grid_operators_on_known_fields():
    x, y = np.linspace(-1, 2, 61), np.linspace(0, 3, 41)
    X, Y = np.meshgrid(x, y)
    dx, dy = x[1] - x[0], y[1] - y[0]
    fx, fy = grad_2d(X**2 * Y, dx, dy)
    assert np.allclose(fx, 2 * X * Y) and np.allclose(fy, X**2)
    assert np.allclose(laplacian_2d(X**2 * Y, dx, dy), 2 * Y)
    assert np.allclose(div_2d(X**2, Y**2, dx, dy), 2 * X + 2 * Y)
    assert np.allclose(curl_z_2d(-0.5 * Y, 0.5 * X, dx, dy), 1.0)   # rigid rotation: 2 Omega


def test_gauss_and_stokes_on_a_rectangle():
    x, y = np.linspace(0, 1, 401), np.linspace(0, 2, 801)
    X, Y = np.meshgrid(x, y)
    dx, dy = x[1] - x[0], y[1] - y[0]
    qx, qy = X**2 * Y, X * Y**3                 # exact answer: 6
    assert np.isclose(boundary_flux_rectangle(qx, qy, x, y), 6.0, rtol=1e-4)
    assert np.isclose(area_integral(div_2d(qx, qy, dx, dy), x, y), 6.0, rtol=1e-4)
    u, v = -Y**3, X**3                          # exact answer: 10
    assert np.isclose(circulation_rectangle(u, v, x, y), 10.0, rtol=1e-4)
    assert np.isclose(area_integral(curl_z_2d(u, v, dx, dy), x, y), 10.0, rtol=1e-4)


def test_exner_rate_sign():
    # flux increasing downstream (divergence > 0) lowers the bed
    assert np.isclose(exner_rate(div_qs=1e-9, porosity=0.4), -1e-9 / 0.6)
    assert exner_rate(div_qs=-1e-9, porosity=0.4) > 0


def test_steady_linear_ridge_and_hilltop_curvature():
    # Exercise 3.5(b): U = 0.1 mm/yr, kappa = 0.005 m^2/yr, L_H = 50 m
    U, kappa = 1e-4, 0.005
    assert np.isclose(-steady_ridge_linear(50.0, U, kappa), 25.0)
    x = np.linspace(-50, 50, 1001)
    curvature = np.gradient(np.gradient(steady_ridge_linear(x, U, kappa), x), x)
    assert np.allclose(curvature[2:-2], -U / kappa)
    assert np.isclose(erosion_rate_from_hilltop_curvature(-U / kappa, kappa), U)


def test_nonlinear_ridge_balances_uplift():
    # at steady state the downslope flux at distance x from the crest equals U x
    U, kappa, S_c = 2e-4, 0.005, 1.25
    x = np.linspace(0, 50, 5001)
    S = -np.gradient(steady_ridge_nonlinear(x, U, kappa, S_c), x)
    q = nonlinear_hillslope_flux(S, kappa, S_c)
    assert np.allclose(q[1:-1], U * x[1:-1], rtol=1e-3, atol=1e-9)
    assert S.max() < S_c


def test_nonlinear_ridge_tends_to_linear_for_large_critical_slope():
    x = np.linspace(0, 50, 11)
    assert np.allclose(steady_ridge_nonlinear(x, 1e-4, 0.005, 1e3),
                       steady_ridge_linear(x, 1e-4, 0.005), rtol=1e-4, atol=1e-9)


def test_flow_divergence_on_ridges_and_valleys():
    # z = S0 y + A cos(k x): analytic div n = A k^2 cos(kx) S0^2 / |grad z|^3
    S0, A, k = 0.2, 5.0, 2 * np.pi / 50
    x, y = np.linspace(0, 100, 401), np.linspace(0, 50, 101)
    X, Y = np.meshgrid(x, y)
    z = S0 * Y + A * np.cos(k * X)
    g = np.sqrt((A * k * np.sin(k * X)) ** 2 + S0**2)
    exact = A * k**2 * np.cos(k * X) * S0**2 / g**3
    numeric = flow_divergence(z, x[1] - x[0], y[1] - y[0])
    assert np.allclose(numeric[2:-2, 2:-2], exact[2:-2, 2:-2], rtol=1e-2, atol=1e-3)
