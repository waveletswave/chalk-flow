import numpy as np
import sympy as sp

from ch05_tools import (EARTH_RADIUS_KM, area_weighted_mean, coriolis_parameter,
                        derivative_in_basis, great_circle_distance, latlon_cell_area,
                        radial_volume, scale_factors_and_unit_vectors)

R = sp.Symbol("R", positive=True)
phi, z = sp.symbols("phi z", real=True)
r, theta = sp.symbols("r theta", positive=True)
CYL = (sp.Matrix([R * sp.cos(phi), R * sp.sin(phi), z]), (R, phi, z))
SPH = (sp.Matrix([r * sp.sin(theta) * sp.cos(phi), r * sp.sin(theta) * sp.sin(phi),
                  r * sp.cos(theta)]), (r, theta, phi))


def test_cylindrical_scale_factors_and_unit_vectors():
    h, e = scale_factors_and_unit_vectors(*CYL)
    assert h == [1, R, 1]
    assert sp.simplify(e[0] - sp.Matrix([sp.cos(phi), sp.sin(phi), 0])) == sp.zeros(3, 1)
    assert sp.simplify(e[1] - sp.Matrix([-sp.sin(phi), sp.cos(phi), 0])) == sp.zeros(3, 1)


def test_spherical_scale_factors_and_orthonormality():
    h, e = scale_factors_and_unit_vectors(*SPH)
    assert sp.simplify(h[0] - 1) == 0 and sp.simplify(h[1] - r) == 0
    assert sp.simplify(h[2] - r * sp.sin(theta)) == 0
    gram = sp.Matrix(3, 3, lambda i, j: sp.simplify(e[i].dot(e[j])))
    assert gram == sp.eye(3)


def test_unit_vector_derivatives():
    _, e = scale_factors_and_unit_vectors(*CYL)
    assert derivative_in_basis(e, CYL[1], 0, 1) == [0, 1, 0]      # de_R/dphi = e_phi
    assert derivative_in_basis(e, CYL[1], 1, 1) == [-1, 0, 0]     # de_phi/dphi = -e_R
    _, e = scale_factors_and_unit_vectors(*SPH)
    q = SPH[1]
    assert derivative_in_basis(e, q, 0, 2) == [0, 0, sp.sin(theta)]
    assert derivative_in_basis(e, q, 1, 2) == [0, 0, sp.cos(theta)]
    assert derivative_in_basis(e, q, 2, 2) == [-sp.sin(theta), -sp.cos(theta), 0]


def test_volumes_of_radial_landforms():
    Rg = np.linspace(0, 5000, 50001)
    cone = radial_volume(1000 * (1 - Rg / 5000), Rg)            # H = 1 km, R_b = 5 km
    assert np.isclose(cone, np.pi * 5000**2 * 1000 / 3, rtol=1e-8)
    Rg = np.linspace(0, 1000, 100001)
    hill = radial_volume(50 * np.exp(-Rg**2 / 100**2), Rg)      # Gaussian hill
    assert np.isclose(hill, np.pi * 100**2 * 50, rtol=1e-8)


def test_fan_volume():
    # Exercise 5.5: sector of 90 degrees, T0 = 5 m, l_d = 2 km -> V = Theta T0 l_d^2
    Rg = np.linspace(0, 60000, 600001)
    V = radial_volume(5 * np.exp(-Rg / 2000), Rg, angle=np.pi / 2)
    assert np.isclose(V, np.pi / 2 * 5 * 2000**2, rtol=1e-6)


def test_latlon_cell_areas():
    assert np.isclose(latlon_cell_area(-0.5, 0.5, 1.0), 12364.15, rtol=1e-6)
    assert np.isclose(latlon_cell_area(35.5, 36.5, 1.0), 10002.81, rtol=1e-6)
    lat = np.arange(-90.0, 90.0, 1.0)
    total = 360 * latlon_cell_area(lat, lat + 1, 1.0).sum()
    assert np.isclose(total, 4 * np.pi * EARTH_RADIUS_KM**2)


def test_area_weighted_mean_removes_high_latitude_bias():
    lat = np.arange(-89.5, 90.0, 1.0)
    field = np.sin(np.radians(lat)) ** 2       # mean over the sphere is exactly 1/3
    assert np.isclose(area_weighted_mean(field, lat, 1.0), 1 / 3, rtol=1e-4)
    assert np.isclose(field.mean(), 0.5, rtol=1e-6)   # equal weights: biased high
    grid = np.tile(field[:, None], (1, 360))
    assert np.isclose(area_weighted_mean(grid, lat, 1.0), 1 / 3, rtol=1e-4)


def test_coriolis_and_great_circles():
    assert coriolis_parameter(0.0) == 0.0
    assert np.isclose(coriolis_parameter(90.0), 1.45842e-4)
    assert np.isclose(coriolis_parameter(-36.0), -coriolis_parameter(36.0))
    assert np.isclose(great_circle_distance(0, 0, 0, 90), np.pi * EARTH_RADIUS_KM / 2)
    assert np.isclose(great_circle_distance(10, 20, -10, -160), np.pi * EARTH_RADIUS_KM)
    assert np.isclose(great_circle_distance(45, 0, 45, 1), 111.195 * np.cos(np.radians(45)),
                      rtol=1e-4)
