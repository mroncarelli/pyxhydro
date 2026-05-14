import json
import os
import numpy as np
import pytest
import xspec as xsp

from astropy import cosmology

from pyxhydro.pygadgetreader import readhead, readsnap
from pyxhydro.gadgetutils.phys_const import keV2K, keV2erg, pi
from pyxhydro.sphprojection.mapping import map2d, specmap
from test_ideal_runs import mapSize

from .randomutils import TrueRandomGenerator, globalRandomSeed

from .__shared import referenceDir, snapshotFile, clear_file, snapshotCenter, testEmissionModel, models_config_file


def vvapec(t: float, met: np.ndarray, redshift: float, nrm: float, abund: str, apecthermal: str, e_min: float,
           e_max: float, nene: int) -> np.ndarray:
    """
    Returns an absorbed (wabs) bapec spectrum.
    """
    xsp.Xset.chatter = 0
    xsp.Xset.addModelString("APECTHERMAL", apecthermal)
    xsp.Xset.abund = abund
    xsp.AllModels.setEnergies(str(e_min) + " " + str(e_max) + " " + str(nene) + " lin")

    pars = {1: t}
    for ind in range(28):
        pars[4 + ind] = met[ind]
    pars[32] = redshift
    pars[33] = nrm
    model = xsp.Model('vvapec', 'test_variable_metallicity', sourceNum=0)
    model.setPars(pars)
    result = np.array(model.values(0))  # [photons s^-1 cm^-2] (already multiplied by norm)
    xsp.AllModels.setEnergies("reset")
    xsp.AllModels -= model.name
    return result


def test_spectrum_from_simulation_matches_reference():
    # Generating random temperature and redshift
    tMin, tMax = 2, 9
    zMin, zMax = 0, 1.5
    TRG = TrueRandomGenerator(globalRandomSeed)
    errMsg = "Random seed: " + str(TRG.initialSeed)  # Assertion error message if test fails
    temp = TRG.uniform(tMin, tMax)  # Gas temperature [keV]
    z = TRG.uniform(zMin, zMax)  # redshift [---]

    # Map parameters
    npix = 50

    # Calculating normalization
    cosmo = cosmology.FlatLambdaCDM(H0=100., Om0=0.3)
    gadget2deg = cosmo.arcsec_per_kpc_comoving(z).to_value() / 3600.  # 1 deg / 1 h^-1 kpc (comoving)
    d_C = 1e3 * cosmo.comoving_distance(z).to_value()  # [h^-1 kpc] comoving
    boxSize = 5100  # [h^-1 kpc] (comoving) box size slightly enlarged to avoid missing particles
    mapSize = boxSize * gadget2deg  # [deg]
    h_Hubble = readhead(snapshotFile, 'hubble')
    map_str = map2d(snapshotFile, 'nenH', 1, center=snapshotCenter, size=mapSize, struct=True, tcut=1e6)
    InenHdl = map_str['map'][0, 0]  # [h^3 cn^-5] (comoving)
    norm = InenHdl * 1e-14 * h_Hubble ** 3 * (1 + z) ** 3 * mapSize ** 2 / (
                4 * pi * d_C ** 2)  # [10^14 cm^-5] (physical)

    # Reading metallcities from the simulation
    mass = readsnap(snapshotFile, 'mass', 'gas')
    Zsim = readsnap(snapshotFile, 'Metallicity', 'gas')[:, 2:]  # [---] Mass fraction, i.e. M_Metal / M_H
    nmet = Zsim.shape[1]
    Zglobal = np.average(Zsim, axis=0, weights=mass)  # [---] Mass fraction, i.e. M_Metal / M_H

    # Reading emission model data
    models_config_file = os.path.join(os.path.dirname(__file__), 'em_reference.json')
    with open(models_config_file) as file:
        json_data = json.load(file)

    # Defining vvapec metallicity array
    metal = np.ndarray(28)

    # Computing reference spectrum
    sp_ref = vvapec(temp, metal, z, 0, norm)  # [photons s^-1 cm^-2] (already multiplied by norm)

    # Creating the spectral map from the snapshot assuming isothermal gas with Gaussian velocity distribution
    sp_map = specmap(snapshotFile, {"name": testEmissionModel}, mapSize, npix, z, center=snapshotCenter,
                     proj='z', tcut=1e6, isothermal=temp * keV2K, novel=True)

    sp = sp_map['data'].sum(axis=(0, 1)) * d_ene * sp_map['pixel_size'] ** 2  # [photons s^-1 cm^2]
    assert sp.sum() == pytest.approx(sp_ref.sum(), rel=1e-4), errMsg
    assert sp == pytest.approx(sp_ref, rel=1e-3), errMsg  # [photons s^-1 cm^2]
