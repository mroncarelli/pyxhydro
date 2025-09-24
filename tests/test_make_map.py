import pytest
import numpy as np
from gadgetutils.phys_const import kpc2cm, m_e, m_p, Xp, Msun2g

import pygadgetreader as pygr
import os

from xraysim.gadgetutils.readspecial import readtemperature, readvelocity
from xraysim.sphprojection.mapping import make_map

dp = np.float64

# Snapshot file on which the tests are performed
snapshotFile = os.environ.get('XRAYSIM') + '/tests/inp/snap_Gadget_sample'

mass = pygr.readsnap(snapshotFile, 'mass', 'gas', units=0, suppress=1)  # [10^10 h^-1 M_Sun]


def test_total_mass():
    """
    The total mass in the projected map must be the same as the snapshot one
    """
    val_snap = np.sum(mass, dtype=dp)  # [10^10 h^-1 M_Sun]
    map_str = make_map(snapshotFile, 'rho', npix=128, struct=True)
    val_map = np.sum(map_str['map'], dtype=dp) * map_str['pixel_size'] ** 2
    assert val_map == pytest.approx(val_snap, rel=1e-6)


def test_int_rho2_over_volume():
    """
    The integral Int(rho^2*dV) in the projected map must be the same as the snapshot one
    """
    rho = pygr.readsnap(snapshotFile, 'rho', 'gas', units=0, suppress=1)  # [10^10 h^2 M_Sun kpc^-3]
    val_snap = np.sum(mass * rho, dtype=dp)  # [10^20 h M_Sun^2 kpc^-3]
    map_str = make_map(snapshotFile, 'rho2', npix=128, struct=True)
    val_map = np.sum(map_str['map'], dtype=dp) * map_str['pixel_size'] ** 2
    assert val_map == pytest.approx(val_snap, rel=1e-6)


def test_total_electron_mass():
    """
    The total electron mass in the projected map must be the same as the snapshot one
    """
    x_e = pygr.readsnap(snapshotFile, 'ne', 'gas', units=0, suppress=1)  # n_e / n_H [---]
    val_snap = np.sum(mass * x_e, dtype=dp) * Xp * m_e / m_p  # Electron mass [10^10 h^-1 M_Sun]
    map_str = make_map(snapshotFile, 'ne', npix=128, struct=True)
    val_map = np.sum(map_str['map'], dtype=dp) * m_e * 1e-10 / Msun2g * (map_str['pixel_size'] * kpc2cm) ** 2
    assert val_map == pytest.approx(val_snap, rel=1e-6)


def test_total_hydrogen_mass():
    """
    The total Hydrogen mass in the projected map must be the same as the snapshot one
    """
    val_snap = np.sum(mass, dtype=dp) * Xp  # Hydrogen mass [10^10 h^-1 M_Sun]
    map_str = make_map(snapshotFile, 'nH', npix=128, struct=True)  # [h cm^-2]
    val_map = np.sum(map_str['map'], dtype=dp) * m_p * 1e-10 / Msun2g * (
                map_str['pixel_size'] * kpc2cm) ** 2  # [10^10 h^-1 M_Sun]
    assert val_map == pytest.approx(val_snap, rel=1e-6)


def test_total_emission_measure():
    """
    The total emission measure Int(rho_e*rho_H*dV) in the projected map must be the same as the snapshot one
    """
    rho = pygr.readsnap(snapshotFile, 'rho', 'gas', units=0, suppress=1)  # [10^10 h^2 M_Sun kpc^-3]
    x_e = pygr.readsnap(snapshotFile, 'ne', 'gas', units=0, suppress=1)  # n_e / n_H [---]
    val_snap = np.sum(mass * rho * x_e, dtype=dp) * Xp ** 2 * m_e / m_p  # [10^20 h M_Sun^2 kpc^-3]
    map_str = make_map(snapshotFile, 'nenH', npix=128, struct=True)  # [h^3 cm^-5]
    val_map = np.sum(map_str['map'], dtype=dp) * m_e * m_p * 1e-20 / Msun2g ** 2 * (
                map_str['pixel_size'] * kpc2cm) ** 2 * kpc2cm ** 3  # [10^20 h M_Sun^2 kpc^-3]
    assert val_map == pytest.approx(val_snap, rel=1e-6)


def test_average_tmw():
    """
    The average n_e-weighted temperature in the projected map must be the same as the snapshot one
    """
    temp = readtemperature(snapshotFile, suppress=1)  # [K]
    x_e = pygr.readsnap(snapshotFile, 'ne', 'gas', units=0, suppress=1)  # n_e / n_H [---]
    val_snap = np.sum(mass * x_e * temp, dtype=dp) / np.sum(mass * x_e, dtype=dp)
    map_str = make_map(snapshotFile, 'Tmw', npix=128, struct=True)
    val_map = np.sum(map_str['map'] * map_str['norm'], dtype=dp) / np.sum(map_str['norm'], dtype=dp)
    assert val_map == pytest.approx(val_snap, rel=1e-6)


def test_average_tew():
    """
    The average emission-weighted (n_e^2) temperature in the projected map must be the same as the snapshot one
    """
    rho = pygr.readsnap(snapshotFile, 'rho', 'gas', units=0, suppress=1)  # [10^10 h^2 M_Sun kpc^-3]
    x_e = pygr.readsnap(snapshotFile, 'ne', 'gas', units=0, suppress=1)  # n_e / n_H [---]
    temp = readtemperature(snapshotFile, suppress=1)  # [K]
    val_snap = np.sum(mass * rho * x_e ** 2 * temp, dtype=dp) / np.sum(mass * rho * x_e ** 2, dtype=dp)
    map_str = make_map(snapshotFile, 'Tew', npix=128, struct=True)
    val_map = np.sum(map_str['map'] * map_str['norm']) / np.sum(map_str['norm'])
    assert val_map == pytest.approx(val_snap, rel=1e-6)


def test_total_electron_momentum():
    """
    The total momentum of free electrons in the projected map must be the same as the snapshot one
    """
    x_e = pygr.readsnap(snapshotFile, 'ne', 'gas', units=0, suppress=1)  # n_e / n_H [---]
    for index in range(3):
        vel = readvelocity(snapshotFile, units='km/s', suppress=1)[:, index]  # [km s^-1]
        val_snap = np.sum(mass * x_e * vel)
        map_str = make_map(snapshotFile, 'vmw', proj=index, npix=128, struct=True)
        val_map = np.sum(map_str['map'] * map_str['norm']) * map_str['pixel_size'] ** 2
        assert val_map == pytest.approx(val_snap, rel=1e-6)


def test_total_ew_momentum():
    """
    The total n_e^2-weighted momentum in the projected map must be the same as the snapshot one
    """
    rho = pygr.readsnap(snapshotFile, 'rho', 'gas', units=0, suppress=1)  # [10^10 h^2 M_Sun kpc^-3]
    x_e = pygr.readsnap(snapshotFile, 'ne', 'gas', units=0, suppress=1)  # n_e / n_H [---]
    for index in range(3):
        vel = readvelocity(snapshotFile, units='km/s', suppress=1)[:, index]  # [km s^-1]
        val_snap = np.sum(mass * rho * x_e ** 2 * vel)
        map_str = make_map(snapshotFile, 'vew', proj=index, npix=128, struct=True)
        val_map = np.sum(map_str['map'] * map_str['norm']) * map_str['pixel_size'] ** 2
        assert val_map == pytest.approx(val_snap, rel=1e-6)


def test_average_electon_velocity_dispersion():
    """
    The average velocity dispersion of free-electrons in the projected map must be the same as the snapshot one
    """
    x_e = pygr.readsnap(snapshotFile, 'ne', 'gas', units=0, suppress=1)  # n_e / n_H [---]
    for index in range(3):
        vel = readvelocity(snapshotFile, units='km/s', suppress=1)[:, index]  # [km s^-1
        val_snap = np.sqrt(
            np.sum(mass * x_e * vel ** 2) / np.sum(mass * x_e) - (np.sum(mass * x_e * vel) / np.sum(mass * x_e)) ** 2)
        map_str = make_map(snapshotFile, 'wmw', proj=index, npix=128, struct=True)
        val1_map = np.sum((map_str['map'] ** 2 + map_str['map2'] ** 2) * map_str['norm']) / np.sum(map_str['norm'])
        val2_map = (np.sum(map_str['map2'] * map_str['norm']) / np.sum(map_str['norm'])) ** 2
        val_map = np.sqrt(val1_map - val2_map)
        assert val_map == pytest.approx(val_snap, rel=1e-6)


def test_average_ew_velocity_dispersion():
    """
    The average n_e^2-weighted velocity dispersion of free electrons in the projected map must be the same as the
    snapshot one
    """
    x_e = pygr.readsnap(snapshotFile, 'ne', 'gas', units=0, suppress=1)  # n_e / n_H [---]
    rho = pygr.readsnap(snapshotFile, 'rho', 'gas', units=0, suppress=1)  # [10^10 h^2 M_Sun kpc^-3]
    for index in range(3):
        vel = readvelocity(snapshotFile, units='km/s', suppress=1)[:, index]  # [km s^-1
        val_snap = np.sqrt(np.sum(mass * rho * x_e ** 2 * vel ** 2) / np.sum(mass * rho * x_e ** 2) - (
                    np.sum(mass * rho * x_e ** 2 * vel) / np.sum(mass * rho * x_e ** 2)) ** 2)
        map_str = make_map(snapshotFile, 'wew', proj=index, npix=128, struct=True)
        val1_map = np.sum((map_str['map'] ** 2 + map_str['map2'] ** 2) * map_str['norm']) / np.sum(map_str['norm'])
        val2_map = (np.sum(map_str['map2'] * map_str['norm']) / np.sum(map_str['norm'])) ** 2
        val_map = np.sqrt(val1_map - val2_map)
        assert val_map == pytest.approx(val_snap, rel=1e-6)
