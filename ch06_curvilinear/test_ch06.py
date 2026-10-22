import numpy as np
import sympy as sp
from scipy.integrate import quad, trapezoid

from ch06_tools import (advect_scalar, convective_acceleration, curl, div,
                        ferguson_church_settling_velocity, grad, laplacian, poiseuille_discharge,
                        poiseuille_velocity, rankine_pressure, rankine_vortex, rouse_number,
                        rouse_profile, steady_dome_profile, stokes_flow_sphere,
                        stokes_settling_velocity, stokes_stream_function, superelevation,
                        theis_drawdown, thiem_head, vector_laplacian)

R = sp.Symbol("R", positive=True)
phi, z = sp.symbols("phi z", real=True)
CYL, H_CYL = (R, phi, z), (1, R, 1)
r, theta = sp.symbols("r theta", positive=True)
SPH, H_SPH = (r, theta, phi), (1, r, r * sp.sin(theta))


def cyl_fields():
    return [sp.Function(n)(R, phi, z) for n in ("u_R", "u_phi", "u_z")]


def test_cylindrical_operators_match_kundu():
    uR, up, uz = u = cyl_fields()
    f = sp.Function("f")(R, phi, z)
    assert sp.simplify(div(u, CYL, H_CYL) - (sp.diff(R * uR, R) / R + sp.diff(up, phi) / R
                                              + sp.diff(uz, z))) == 0
    lap_f = sp.diff(R * sp.diff(f, R), R) / R + sp.diff(f, phi, 2) / R**2 + sp.diff(f, z, 2)
    assert sp.simplify(laplacian(f, CYL, H_CYL) - lap_f) == 0
    assert sp.simplify(curl(u, CYL, H_CYL)[2] - (sp.diff(R * up, R) - sp.diff(uR, phi)) / R) == 0
    VL = vector_laplacian(u, CYL, H_CYL)
    lap = lambda g: laplacian(g, CYL, H_CYL)
    assert sp.simplify(VL[0] - (lap(uR) - uR / R**2 - 2 / R**2 * sp.diff(up, phi))) == 0
    assert sp.simplify(VL[1] - (lap(up) + 2 / R**2 * sp.diff(uR, phi) - up / R**2)) == 0
    assert sp.simplify(VL[2] - lap(uz)) == 0


def test_spherical_divergence_and_laplacian():
    ur, ut, up = [sp.Function(n)(r, theta, phi) for n in ("u_r", "u_theta", "u_phi")]
    s = sp.sin(theta)
    kundu = (sp.diff(r**2 * ur, r) / r**2 + sp.diff(ut * s, theta) / (r * s)
             + sp.diff(up, phi) / (r * s))
    assert sp.simplify(div([ur, ut, up], SPH, H_SPH) - kundu) == 0
    f = sp.Function("f")(r, theta, phi)
    kundu = (sp.diff(r**2 * sp.diff(f, r), r) / r**2 + sp.diff(s * sp.diff(f, theta), theta)
             / (r**2 * s) + sp.diff(f, phi, 2) / (r**2 * s**2))
    assert sp.simplify(laplacian(f, SPH, H_SPH) - kundu) == 0


def test_convective_acceleration_extra_terms():
    uR, up, uz = u = cyl_fields()
    a = convective_acceleration(u, CYL, H_CYL)
    assert sp.simplify(a[0] - (advect_scalar(uR, u, CYL, H_CYL) - up**2 / R)) == 0
    assert sp.simplify(a[1] - (advect_scalar(up, u, CYL, H_CYL) + uR * up / R)) == 0
    assert sp.simplify(a[2] - advect_scalar(uz, u, CYL, H_CYL)) == 0


def test_uniform_flow_has_no_acceleration_in_cylindrical_components():
    # Exercise 6.2: u = U e_x -> u_R = U cos(phi), u_phi = -U sin(phi)
    U = sp.Symbol("U", positive=True)
    u = [U * sp.cos(phi), -U * sp.sin(phi), 0]
    assert [sp.simplify(c) for c in convective_acceleration(u, CYL, H_CYL)] == [0, 0, 0]


def test_stokes_flow_satisfies_stokes_equations():
    a, U, mu = sp.symbols("a U mu", positive=True)
    s, c = sp.sin(theta), sp.cos(theta)
    u = [U * c * (1 - 3 * a / (2 * r) + a**3 / (2 * r**3)),
         -U * s * (1 - 3 * a / (4 * r) - a**3 / (4 * r**3)), 0]
    p = -sp.Rational(3, 2) * mu * a * U * c / r**2
    assert sp.simplify(div(u, SPH, H_SPH)) == 0
    VL, gp = vector_laplacian(u, SPH, H_SPH), grad(p, SPH, H_SPH)
    assert all(sp.simplify(gp[i] - mu * VL[i]) == 0 for i in range(3))
    assert sp.simplify(curl(u, SPH, H_SPH)[2] + sp.Rational(3, 2) * U * a * s / r**2) == 0


def test_stream_function_matches_velocity():
    a, U, th, rr, eps = 1.0, 2.0, 0.7, 1.8, 1e-6
    ur, ut = stokes_flow_sphere(rr, th, a, U)
    dpsi_dth = (stokes_stream_function(rr, th + eps, a, U)
                - stokes_stream_function(rr, th - eps, a, U)) / (2 * eps)
    dpsi_dr = (stokes_stream_function(rr + eps, th, a, U)
               - stokes_stream_function(rr - eps, th, a, U)) / (2 * eps)
    assert np.isclose(ur, dpsi_dth / (rr**2 * np.sin(th)), rtol=1e-6)
    assert np.isclose(ut, -dpsi_dr / (rr * np.sin(th)), rtol=1e-6)


def test_rankine_dust_devil():
    # Exercise 6.3: a = 5 m, U_m = 15 m/s, air
    a, Um, rho = 5.0, 15.0, 1.2
    assert np.isclose(2 * np.pi * a * rankine_vortex(a, a, Um), 471.24, rtol=1e-4)
    assert np.isclose(rankine_pressure(0.0, a, Um, rho), -rho * Um**2)        # -270 Pa
    deficit = quad(lambda s: rho * rankine_vortex(s, a, Um) ** 2 / s, 1e-12, a)[0] \
        + quad(lambda s: rho * rankine_vortex(s, a, Um) ** 2 / s, a, np.inf)[0]
    assert np.isclose(deficit, 270.0, rtol=1e-6)


def test_poiseuille_flow_in_a_macropore():
    # Exercise 6.5: a = 0.5 mm, unit hydraulic gradient G = rho g, water
    a, G, mu = 5e-4, 1000 * 9.81, 1.0e-3
    Q = poiseuille_discharge(a, G, mu)
    Rg = np.linspace(0, a, 20001)
    assert np.isclose(trapezoid(poiseuille_velocity(Rg, a, G, mu) * 2 * np.pi * Rg, Rg), Q)
    mean = Q / (np.pi * a**2)
    assert np.isclose(mean, poiseuille_velocity(0, a, G, mu) / 2)
    assert np.isclose(Q, 2.408e-7, rtol=1e-3) and np.isclose(mean, 0.3066, rtol=1e-3)


def test_settling_velocities_and_rouse_numbers():
    d = 30e-6
    assert np.isclose(stokes_settling_velocity(d), 8.09e-4, rtol=1e-3)
    assert np.isclose(ferguson_church_settling_velocity(d), 7.84e-4, rtol=1e-3)
    assert np.isclose(ferguson_church_settling_velocity(1e-7), stokes_settling_velocity(1e-7),
                      rtol=1e-3)
    assert np.isclose(rouse_number(8.09e-4, 0.05), 0.0395, rtol=1e-2)   # silt: wash load
    assert np.isclose(rouse_number(0.023, 0.05), 1.12, rtol=1e-2)       # sand: near the bed
    z = np.linspace(0.05, 0.99, 50)
    c = rouse_profile(z, 1.0, 0.05, 1.0)
    assert np.isclose(c[0], 1.0) and np.all(np.diff(c) < 0)


def test_superelevation_wells_and_domes():
    assert np.isclose(superelevation(1.5, 40.0, 120.0), 0.0765, rtol=1e-3)   # Exercise 6.6(b)
    # late-time Theis drawdown differences approach the Thiem logarithm
    Q, T, S = 0.01, 1e-3, 1e-4
    ds = theis_drawdown(10.0, 1e7, Q, T, S) - theis_drawdown(100.0, 1e7, Q, T, S)
    dh = thiem_head(100.0, 0.1, 0.0, Q, T) - thiem_head(10.0, 0.1, 0.0, Q, T)
    assert np.isclose(ds, dh, rtol=1e-3)
    # the Theis solution obeys S ds/dt = T (1/r) d/dr (r ds/dr)
    rr, t, e = 30.0, 5e4, 1e-3
    s = lambda rv, tv: theis_drawdown(rv, tv, Q, T, S)
    lhs = S * (s(rr, t * (1 + e)) - s(rr, t * (1 - e))) / (2 * t * e)
    flux = lambda rv: rv * (s(rv + rr * e, t) - s(rv - rr * e, t)) / (2 * rr * e)
    rhs = T * (flux(rr * (1 + e)) - flux(rr * (1 - e))) / (2 * rr * e) / rr
    assert np.isclose(lhs, rhs, rtol=1e-4)
    # Exercise 6.8(b): dome relief 12.5 m, half that of the 25 m ridge
    assert np.isclose(-steady_dome_profile(50.0, 1e-4, 0.005), 12.5)
