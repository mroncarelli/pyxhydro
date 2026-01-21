"""
The tests in this file have parameters that are chosen randomly with a true random generator: this means that for every
run these parameters change. The reproducibility is assured by the initial random seed that is shown in the error
message in case of test failure: in order to reproduce the error one must take not of the seed (i.e. 12345678) and call

pytest --seed 12345678
"""

import pytest

from pyxhydro.specutils import absorption as spabs
from pyxhydro.specutils import tables
import numpy as np

import xspec as xsp
from .randomutils import TrueRandomGenerator, globalRandomSeed
from .__shared import *

specTable = tables.read_spectable(referenceSpecTableFile)

xsp.Xset.chatter = 0
xsp.Xset.addModelString("APECTHERMAL", "yes")
if specTable.get("abund"):
    xsp.Xset.abund = specTable.get("abund")
energyTable = specTable.get("energy")
nz, nt, nene = specTable.get("data").shape
dE = energyTable[1] - energyTable[0]
e_min = energyTable[0] - 0.5 * dE
e_max = energyTable[-1] + 0.5 * dE

# Here I use this method to generate some true random numbers to differentiate the tests.
TRG = TrueRandomGenerator(globalRandomSeed)
errMsg = "Random seed: " + str(TRG.initialSeed)  # Assertion error message if test fails

nH_min, nH_max = 0., 0.1

def test_table_values_decrease_exponentially_with_nh(delta_nh=0.02):
    if 'nh' in specTable:
        nh_old = specTable.get('nh')  # [10^22 cm^-2]
    else:
        nh_old = 0.

    nh_new = nh_old + delta_nh
    spectable_new = spabs.convert_nh(specTable, nh_new)
    energy = specTable.get('energy')
    expected_ratio = np.exp(-delta_nh * spabs.sigma_abs_galactic(energy))
    nz, nt = specTable.get('data').shape[0:2]
    for iz in range(nz):
        for it in range(nt):
            assert specTable.get('data')[iz, it, :] * expected_ratio == pytest.approx(
                spectable_new.get('data')[iz, it, :])


def wabs_apec_spectrum(nh, kt, z, metal, norm):
    xsp.AllModels.setEnergies(str(e_min) + " " + str(e_max) + " " + str(nene) + " lin")
    pars = {}
    pars[1] = float(nh)  # [10^22 cm^-2]
    pars[2] = float(kt)  # [keV]
    for ind in range(28):
        pars[5 + ind] = float(metal)
    pars[33] = float(z)
    pars[34] = float(norm)
    model = xsp.Model('wabs(vvapec)', 'test_convert_nh', 0)
    model.setPars(pars)
    result = np.array(model.values(0), dtype=np.float32)  # [photons s^-1 cm^-2]
    xsp.AllModels.setEnergies("reset")
    xsp.AllModels -= model.name
    return result


def test_spectrum_from_table_with_convert_nh_must_match_pyxspec():
    """
    Converting a spectrum from a table with the convert_nh method must match the corresponding spectrum computed
    directly using the wabs(vvapec) model with PyXspec.
    """
    metal = specTable.get('metallicity')
    iz = TRG.int(0, nz)
    it = TRG.int(0, nt)
    nh = TRG.uniform(nH_min, nH_max)
    spectrum_ref = wabs_apec_spectrum(nh, specTable.get("temperature")[it], specTable.get("z")[iz], metal, 1)
    spectrum_calc = spabs.convert_nh(specTable, nh).get("data")[iz, it, :]
    assert spectrum_calc == pytest.approx(spectrum_ref, rel=1e-3), errMsg
