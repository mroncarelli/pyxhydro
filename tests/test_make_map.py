"""
This file contains tests designed to test the mapping procedures. Some tests require an alpha parameter. or an array of
alpha parameters, amd their values are chosen randomly with a true random generator: this means that for every
run the parameters change. The reproducibility is assured by the initial random seed that is shown in the error message
in case of test failure: in order to reproduce the error one must take not of the seed (i.e. 12345678) and call

pytest --seed 12345678
"""

import pytest
import numpy as np
from gadgetutils.phys_const import kpc2cm, m_e, m_p, Xp, Msun2g

import pygadgetreader as pygr
import os

from xraysim.gadgetutils.readspecial import readtemperature, readvelocity
from xraysim.sphprojection.mapping import make_map
from .randomutils import TrueRandomGenerator, globalRandomSeed

DP = np.float64

# Number of pixels
npix = 128

# Relative tolerance (some test with alpha-weight may fail with 1e-6)
relTol = 5e-6

# Snapshot file on which the tests are performed
snapshotFile = os.environ.get('XRAYSIM') + '/tests/inp/snap_Gadget_sample'

# Mass must be read by all tests
mass = pygr.readsnap(snapshotFile, 'mass', 'gas', units=0, suppress=1)  # [10^10 h^-1 M_Sun]

# Here I use this method to generate some true random numbers to differentiate the tests.
seed = int.from_bytes(os.urandom(4))  # 4-bytes int generated with a true random function
rs = np.random.RandomState(seed)  # Initialization of the random state

alphaMin = -2
alphaMax = 2
TRG = TrueRandomGenerator(globalRandomSeed)
errMsg = "Random seed: " + str(TRG.initialSeed)  # Assertion error message if test fails
alpha = TRG.uniform(alphaMin, alphaMax)  # randomly generated vale of alpha
alpha_vec = np.asarray([TRG.uniform(alphaMin, alphaMax), TRG.uniform(alphaMin, alphaMax),
                        TRG.uniform(alphaMin, alphaMax)])


def test_total_mass():
    """
    The total mass in the projected map must be the same as the snapshot one
    """
    val_snap = np.sum(mass, dtype=DP)  # [10^10 h^-1 M_Sun]
    map_str = make_map(snapshotFile, 'rho', npix=npix, struct=True)
    val_map = np.sum(map_str['map'], dtype=DP) * map_str['pixel_size'] ** 2
    assert val_map == pytest.approx(val_snap, rel=relTol)


def test_int_rho2_over_volume():
    """
    The integral Int(rho^2*dV) in the projected map must be the same as the snapshot one
    """
    rho = pygr.readsnap(snapshotFile, 'rho', 'gas', units=0, suppress=1)  # [10^10 h^2 M_Sun kpc^-3]
    val_snap = np.sum(mass * rho, dtype=DP)  # [10^20 h M_Sun^2 kpc^-3]
    map_str = make_map(snapshotFile, 'rho2', npix=npix, struct=True)
    val_map = np.sum(map_str['map'], dtype=DP) * map_str['pixel_size'] ** 2
    assert val_map == pytest.approx(val_snap, rel=relTol)


def test_total_electron_mass():
    """
    The total electron mass in the projected map must be the same as the snapshot one
    """
    x_e = pygr.readsnap(snapshotFile, 'ne', 'gas', units=0, suppress=1)  # n_e / n_H [---]
    val_snap = np.sum(mass * x_e, dtype=DP) * Xp * m_e / m_p  # Electron mass [10^10 h^-1 M_Sun]
    map_str = make_map(snapshotFile, 'ne', npix=npix, struct=True)
    val_map = np.sum(map_str['map'], dtype=DP) * m_e * 1e-10 / Msun2g * (map_str['pixel_size'] * kpc2cm) ** 2
    assert val_map == pytest.approx(val_snap, rel=relTol)


def test_total_hydrogen_mass():
    """
    The total Hydrogen mass in the projected map must be the same as the snapshot one
    """
    val_snap = np.sum(mass, dtype=DP) * Xp  # Hydrogen mass [10^10 h^-1 M_Sun]
    map_str = make_map(snapshotFile, 'nH', npix=npix, struct=True)  # [h cm^-2]
    val_map = np.sum(map_str['map'], dtype=DP) * m_p * 1e-10 / Msun2g * (
            map_str['pixel_size'] * kpc2cm) ** 2  # [10^10 h^-1 M_Sun]
    assert val_map == pytest.approx(val_snap, rel=relTol)


def test_total_emission_measure():
    """
    The total emission measure Int(rho_e*rho_H*dV) in the projected map must be the same as the snapshot one
    """
    rho = pygr.readsnap(snapshotFile, 'rho', 'gas', units=0, suppress=1)  # [10^10 h^2 M_Sun kpc^-3]
    x_e = pygr.readsnap(snapshotFile, 'ne', 'gas', units=0, suppress=1)  # n_e / n_H [---]
    val_snap = np.sum(mass * rho * x_e, dtype=DP) * Xp ** 2 * m_e / m_p  # [10^20 h M_Sun^2 kpc^-3]
    map_str = make_map(snapshotFile, 'nenH', npix=npix, struct=True)  # [h^3 cm^-5]
    val_map = np.sum(map_str['map'], dtype=DP) * m_e * m_p * 1e-20 / Msun2g ** 2 * (
            map_str['pixel_size'] * kpc2cm) ** 2 * kpc2cm ** 3  # [10^20 h M_Sun^2 kpc^-3]
    assert val_map == pytest.approx(val_snap, rel=relTol)


def test_average_tmw():
    """
    The average n_e-weighted temperature in the projected map must be the same as the snapshot one
    """
    temp = readtemperature(snapshotFile, suppress=1)  # [K]
    x_e = pygr.readsnap(snapshotFile, 'ne', 'gas', units=0, suppress=1)  # n_e / n_H [---]
    val_snap = np.sum(mass * x_e * temp, dtype=DP) / np.sum(mass * x_e, dtype=DP)
    map_str = make_map(snapshotFile, 'Tmw', npix=npix, struct=True)
    val_map = np.sum(map_str['map'] * map_str['norm'], dtype=DP) / np.sum(map_str['norm'], dtype=DP)
    assert val_map == pytest.approx(val_snap, rel=relTol)


def test_average_tew():
    """
    The average emission-weighted (n_e^2) temperature in the projected map must be the same as the snapshot one
    """
    rho = pygr.readsnap(snapshotFile, 'rho', 'gas', units=0, suppress=1)  # [10^10 h^2 M_Sun kpc^-3]
    x_e = pygr.readsnap(snapshotFile, 'ne', 'gas', units=0, suppress=1)  # n_e / n_H [---]
    temp = readtemperature(snapshotFile, suppress=1)  # [K]
    val_snap = np.sum(mass * rho * x_e ** 2 * temp, dtype=DP) / np.sum(mass * rho * x_e ** 2, dtype=DP)
    map_str = make_map(snapshotFile, 'Tew', npix=npix, struct=True)
    val_map = np.sum(map_str['map'] * map_str['norm'], dtype=DP) / np.sum(map_str['norm'], dtype=DP)
    assert val_map == pytest.approx(val_snap, rel=relTol)


def test_average_tsl():
    """
    The average spectroscopic-like (n_e^2*T^-0.75) temperature in the projected map must be the same as the snapshot one
    """
    rho = pygr.readsnap(snapshotFile, 'rho', 'gas', units=0, suppress=1)  # [10^10 h^2 M_Sun kpc^-3]
    x_e = pygr.readsnap(snapshotFile, 'ne', 'gas', units=0, suppress=1)  # n_e / n_H [---]
    temp = readtemperature(snapshotFile, suppress=1)  # [K]
    val_snap = (np.sum(mass * rho * x_e ** 2 * temp ** 0.25, dtype=DP) /
                np.sum(mass * rho * x_e ** 2 * temp ** -0.75, dtype=DP))
    map_str = make_map(snapshotFile, 'Tsl', npix=npix, struct=True)
    val_map = np.sum(map_str['map'] * map_str['norm'], dtype=DP) / np.sum(map_str['norm'], dtype=DP)
    assert val_map == pytest.approx(val_snap, rel=relTol)


def test_average_taw():
    """
    The average alpha-weighted (n_e^2*T^alpha) temperature in the projected map must be the same as the snapshot one
    """
    rho = pygr.readsnap(snapshotFile, 'rho', 'gas', units=0, suppress=1)  # [10^10 h^2 M_Sun kpc^-3]
    x_e = pygr.readsnap(snapshotFile, 'ne', 'gas', units=0, suppress=1)  # n_e / n_H [---]
    temp = readtemperature(snapshotFile, suppress=1)  # [K]
    val_snap = (np.sum(mass * rho * x_e ** 2 * temp ** (alpha + 1), dtype=DP) /
                np.sum(mass * rho * x_e ** 2 * temp ** alpha, dtype=DP))
    map_str = make_map(snapshotFile, 'Taw', npix=npix, alpha=alpha, struct=True)
    val_map = np.sum(map_str['map'] * map_str['norm'], dtype=DP) / np.sum(map_str['norm'], dtype=DP)
    assert val_map == pytest.approx(val_snap, rel=relTol), errMsg


def test_taw_with_alpha_vector_matches_scalar():
    """
    Computing alpha-weighted temperature with alpha vector must match the corresponding ones computed in scalar mode.
    """
    for iproj in range(3):
        map_alpha_vec = make_map(snapshotFile, 'taw', proj=iproj, npix=npix, alpha=alpha_vec, struct=True)
        for index, alpha_scalar in enumerate(alpha_vec):
            map_alpha = make_map(snapshotFile, 'taw', proj=iproj, npix=npix, alpha=alpha_scalar, struct=True)
            for key in map_alpha_vec:
                if key not in ["map", "norm", "alpha"]:
                    assert map_alpha_vec[key] == pytest.approx(map_alpha[key], rel=relTol), errMsg
            assert map_alpha_vec["map"][:, :, index] == pytest.approx(map_alpha["map"], rel=relTol), errMsg
            assert map_alpha_vec["norm"][:, :, index] == pytest.approx(map_alpha["norm"], rel=relTol), errMsg
            assert alpha_scalar == map_alpha["alpha"], errMsg


def test_total_electron_momentum():
    """
    The total momentum of free electrons in the projected map must be the same as the snapshot one
    """
    x_e = pygr.readsnap(snapshotFile, 'ne', 'gas', units=0, suppress=1)  # n_e / n_H [---]
    for iproj in range(3):
        vel = readvelocity(snapshotFile, units='km/s', suppress=1)[:, iproj]  # [km s^-1]
        val_snap = np.sum(mass * x_e * vel, dtype=DP)
        map_str = make_map(snapshotFile, 'vmw', proj=iproj, npix=npix, struct=True)
        val_map = np.sum(map_str['map'] * map_str['norm'], dtype=DP) * map_str['pixel_size'] ** 2
        assert val_map == pytest.approx(val_snap, rel=relTol)


def test_total_ew_momentum():
    """
    The total n_e^2-weighted momentum in the projected map must be the same as the snapshot one
    """
    rho = pygr.readsnap(snapshotFile, 'rho', 'gas', units=0, suppress=1)  # [10^10 h^2 M_Sun kpc^-3]
    x_e = pygr.readsnap(snapshotFile, 'ne', 'gas', units=0, suppress=1)  # n_e / n_H [---]
    for iproj in range(3):
        vel = readvelocity(snapshotFile, units='km/s', suppress=1)[:, iproj]  # [km s^-1]
        val_snap = np.sum(mass * rho * x_e ** 2 * vel, dtype=DP)
        map_str = make_map(snapshotFile, 'vew', proj=iproj, npix=npix, struct=True)
        val_map = np.sum(map_str['map'] * map_str['norm'], dtype=DP) * map_str['pixel_size'] ** 2
        assert val_map == pytest.approx(val_snap, rel=relTol)


def test_total_aw_momentum():
    """
    The total alpha-weighted (n_e^2*T^alpha) momentum in the projected map must be the same as the snapshot one
    """
    rho = pygr.readsnap(snapshotFile, 'rho', 'gas', units=0, suppress=1)  # [10^10 h^2 M_Sun kpc^-3]
    x_e = pygr.readsnap(snapshotFile, 'ne', 'gas', units=0, suppress=1)  # n_e / n_H [---]
    temp = readtemperature(snapshotFile, suppress=1)  # [K]
    for iproj in range(3):
        vel = readvelocity(snapshotFile, units='km/s', suppress=1)[:, iproj]  # [km s^-1]
        val_snap = np.sum(mass * rho * x_e ** 2 * temp ** alpha * vel, dtype=DP)
        map_str = make_map(snapshotFile, 'vaw', proj=iproj, npix=npix, alpha=alpha, struct=True)
        val_map = np.sum(map_str['map'] * map_str['norm'], dtype=DP) * map_str['pixel_size'] ** 2
        assert val_map == pytest.approx(val_snap, rel=relTol), errMsg


def test_vaw_with_alpha_vector_matches_scalar():
    """
    Computing alpha-weighted velocity with alpha vector must match the corresponding ones computed in scalar mode.
    """
    for iproj in range(3):
        map_alpha_vec = make_map(snapshotFile, 'vaw', proj=iproj, npix=npix, alpha=alpha_vec, struct=True)
        for index, alpha_scalar in enumerate(alpha_vec):
            map_alpha = make_map(snapshotFile, 'vaw', proj=iproj, npix=npix, alpha=alpha_scalar, struct=True)
            for key in map_alpha_vec:
                if key not in ["map", "norm", "alpha"]:
                    assert map_alpha_vec[key] == pytest.approx(map_alpha[key], rel=relTol), errMsg
            # Here the difference must be tested in absolute value (km/s) since values may be negative or ~0
            assert map_alpha_vec["map"][:, :, index] == pytest.approx(map_alpha["map"], abs=1e-3), errMsg
            assert map_alpha_vec["norm"][:, :, index] == pytest.approx(map_alpha["norm"], rel=relTol), errMsg
            assert alpha_scalar == map_alpha["alpha"], errMsg


def test_average_electron_velocity_dispersion():
    """
    The average velocity dispersion of free-electrons in the projected map must be the same as the snapshot one
    """
    x_e = pygr.readsnap(snapshotFile, 'ne', 'gas', units=0, suppress=1)  # n_e / n_H [---]
    nsum = np.sum(mass * x_e, dtype=DP)  # [10^10 h^-1 M_Sun]
    for iproj in range(3):
        vel = readvelocity(snapshotFile, units='km/s', suppress=1)[:, iproj]  # [km s^-1]
        val_snap = np.sqrt(
            np.sum(mass * x_e * vel ** 2) / nsum - (np.sum(mass * x_e * vel) / nsum) ** 2)
        map_str = make_map(snapshotFile, 'wmw', proj=iproj, npix=npix, struct=True)
        val1_map = (np.sum((map_str['map'] ** 2 + map_str['map2'] ** 2) * map_str['norm'], dtype=DP) /
                    np.sum(map_str['norm'], dtype=DP))
        val2_map = (np.sum(map_str['map2'] * map_str['norm'], dtype=DP) / np.sum(map_str['norm'], dtype=DP)) ** 2
        val_map = np.sqrt(val1_map - val2_map)
        assert val_map == pytest.approx(val_snap, rel=relTol)


def test_average_ew_velocity_dispersion():
    """
    The average n_e^2-weighted velocity dispersion of free electrons in the projected map must be the same as the
    snapshot one
    """
    x_e = pygr.readsnap(snapshotFile, 'ne', 'gas', units=0, suppress=1)  # n_e / n_H [---]
    rho = pygr.readsnap(snapshotFile, 'rho', 'gas', units=0, suppress=1)  # [10^10 h^2 M_Sun kpc^-3]
    nsum = np.sum(mass * rho * x_e ** 2, dtype=DP)  # [10^20 h M_Sun^2 kpc^-3]
    for iproj in range(3):
        vel = readvelocity(snapshotFile, units='km/s', suppress=1)[:, iproj]  # [km s^-1]
        val_snap = np.sqrt(np.sum(mass * rho * x_e ** 2 * vel ** 2, dtype=DP) / nsum - (
                np.sum(mass * rho * x_e ** 2 * vel, dtype=DP) / nsum) ** 2)
        map_str = make_map(snapshotFile, 'wew', proj=iproj, npix=npix, struct=True)
        val1_map = (np.sum((map_str['map'] ** 2 + map_str['map2'] ** 2) * map_str['norm'], dtype=DP) /
                    np.sum(map_str['norm'], dtype=DP))
        val2_map = (np.sum(map_str['map2'] * map_str['norm'], dtype=DP) / np.sum(map_str['norm'], dtype=DP)) ** 2
        val_map = np.sqrt(val1_map - val2_map)
        assert val_map == pytest.approx(val_snap, rel=relTol)


def test_average_aw_velocity_dispersion():
    """
    The average alpha-weighted (n_e^2*T^alpha) velocity dispersion of free electrons in the projected map must be the
    same as the snapshot one
    """
    x_e = pygr.readsnap(snapshotFile, 'ne', 'gas', units=0, suppress=1)  # n_e / n_H [---]
    rho = pygr.readsnap(snapshotFile, 'rho', 'gas', units=0, suppress=1)  # [10^10 h^2 M_Sun kpc^-3]
    temp = readtemperature(snapshotFile, suppress=1)  # [K]
    nsum = np.sum(mass * rho * x_e ** 2 * temp ** alpha, dtype=DP)  # [10^20 h M_Sun^2 kpc^-3 K^alpha]
    for iproj in range(3):
        vel = readvelocity(snapshotFile, units='km/s', suppress=1)[:, iproj]  # [km s^-1
        val_snap = np.sqrt(np.sum(mass * rho * x_e ** 2 * temp ** alpha * vel ** 2, dtype=DP) / nsum -
                           (np.sum(mass * rho * x_e ** 2 * temp ** alpha * vel, dtype=DP) / nsum) ** 2)
        map_str = make_map(snapshotFile, 'waw', proj=iproj, npix=npix, alpha=alpha, struct=True)
        val1_map = (np.sum((map_str['map'] ** 2 + map_str['map2'] ** 2) * map_str['norm'], dtype=DP) /
                    np.sum(map_str['norm'], dtype=DP))
        val2_map = (np.sum(map_str['map2'] * map_str['norm'], dtype=DP) / np.sum(map_str['norm'], dtype=DP)) ** 2
        val_map = np.sqrt(val1_map - val2_map)
        assert val_map == pytest.approx(val_snap, rel=relTol), errMsg


def test_waw_with_alpha_vector_matches_scalar():
    """
    Computing alpha-weighted velocity dispersion with alpha vector must match the corresponding ones computed in scalar
    mode.
    """
    for iproj in range(3):
        map_alpha_vec = make_map(snapshotFile, 'waw', proj=iproj, npix=npix, alpha=alpha_vec, struct=True)
        for index, alpha_scalar in enumerate(alpha_vec):
            map_alpha = make_map(snapshotFile, 'waw', proj=iproj, npix=npix, alpha=alpha_scalar, struct=True)
            for key in map_alpha_vec:
                if key not in ["map", "map2", "norm", "alpha"]:
                    assert map_alpha_vec[key] == pytest.approx(map_alpha[key], rel=relTol), errMsg
            # The check for the velocity dispersion map must be done more carefully as small values may differ
            # due to the roundings.
            map1 = map_alpha_vec["map"][:, :, index]
            map0 = map_alpha["map"]
            for ii in range(npix):
                for jj in range(npix):
                    val0 = map0[ii, jj]
                    val1 = map1[ii, jj]
                    # Testing the difference also in absolute value (km/s) since values may be negative or ~0
                    assert val1 == pytest.approx(val0, abs=2) or val1 == pytest.approx(val0, rel=1e-4), errMsg
            assert map_alpha_vec["map2"][:, :, index] == pytest.approx(map_alpha["map2"], abs=1e-3), errMsg
            assert map_alpha_vec["norm"][:, :, index] == pytest.approx(map_alpha["norm"], rel=relTol), errMsg
            assert alpha_scalar == map_alpha["alpha"], errMsg
