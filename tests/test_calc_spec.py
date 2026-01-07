"""
The tests in this file have parameters that are chosen randomly with a true random generator: this means that for every
run these parameters change. The reproducibility is assured by the initial random seed that is shown in the error
message in case of test failure: in order to reproduce the error one must take not of the seed (i.e. 12345678) and call

pytest --seed 12345678
"""

import pytest

from xraysim.specutils.tables import read_spectable, calc_spec
from .randomutils import TrueRandomGenerator, globalRandomSeed
from .__shared import referenceSpecTableFile

specTable = read_spectable(referenceSpecTableFile)
nz = len(specTable['z'])
nt = len(specTable['temperature'])
zMin, zMax = specTable.get('z').min(), specTable.get('z').max()
tMin, tMax = specTable.get('temperature').min(), specTable.get('temperature').max()

# Here I use this method to generate some true random numbers to differentiate the tests.
TRG = TrueRandomGenerator(globalRandomSeed)
errMsg = "Random seed: " + str(TRG.initialSeed)  # Assertion error message if test fails


@pytest.mark.filterwarnings("ignore")
def test_table_values():
    # A spectrum computed at a z and temperature that match table values must return the values of the table
    for iz in range(1, nz-1):  # no lower and upper bound to avoid getting all zeros
        for it in range(1, nt):    # no lower bound to avoid getting all zeros (upper bound is ok)
            spec = calc_spec(specTable, specTable.get('z')[iz], specTable.get('temperature')[it])
            assert all(val == pytest.approx(ref) for val, ref in zip(spec, specTable.get('data')[iz, it, :])), (
                    "iz = " + str(iz) + ", it = " + str(it))


@pytest.mark.filterwarnings("ignore")
def test_spectra_with_z_smaller_than_table_min_must_be_all_zeros():
    # Spectra of any temperature with z smaller than the table minimum must contain all zeros if no_z_interp is True.
    temp = TRG.uniform(tMin, tMax)  # [keV]
    spec = calc_spec(specTable, specTable.get('z').min() - 1e-3, temp, no_z_interp=True)
    assert spec.all() == 0, errMsg + ", Temperature = " + str(temp)


@pytest.mark.filterwarnings("ignore")
def test_spectra_with_z_larger_than_table_min_must_be_all_zeros():
    # Spectra of any temperature with z larger than the table minimum must contain all zeros if no_z_interp is True.
    temp = TRG.uniform(tMin, tMax)  # [keV]
    spec = calc_spec(specTable, specTable.get('z').max() + 1e-3, temp, no_z_interp=True)
    assert spec.all() == 0, errMsg + ", Temperature = " + str(temp)


@pytest.mark.filterwarnings("ignore")
def test_spectra_with_temperature_smaller_than_table_min_must_be_all_zeros():
    # Spectra of any z with temperature smaller than the table minimum must contain all zeros.
    z = TRG.uniform(zMin, zMax)  # [---]
    spec1 = calc_spec(specTable, z, 0.999 * tMin, no_z_interp=True)
    assert spec1.all() == 0, errMsg + ", Redshift = " + str(z)
    spec2 = calc_spec(specTable, z, 0.999 * tMin, no_z_interp=False)
    assert spec2.all() == 0, errMsg + ", Redshift = " + str(z)


# TODO Refine this test: flat smoothing is not enough, Gaussian smoothing might work
# @pytest.mark.filterwarnings("ignore")
# def test_calc_spec_with_z_from_table_must_match_the_one_computed_with_pyxspec():
#     # A spectrum with redshift corresponding to a table value must match the one computed with PyXspec (vvapec) with
#     # the same parameters of the table.
#     iz = rs.randint(nz)
#     z = float(specTable.get("z")[iz])
#     temp = float(rs.random() * (tMax - tMin) + tMin)  # [keV]
#     energy = specTable.get("energy")
#     de = energy[1] - energy[0]  # [keV] Assumes uniform energy
#     emin, emax = energy[0] - 0.5 * de, energy[-1] + 0.5 * de  # [keV]
#     abund = specTable.get("abund")
#     metal = float(specTable.get("metallicity"))  # [Solar]
#     tbroad = specTable.get("tbroad")
#
#     # Saving current PyXspec settings to restore them at the end of the procedure
#     chatter_ = xsp.Xset.chatter
#     model_strings_ = xsp.Xset.modelStrings
#     abund_ = xsp.Xset.abund[0:4]
#
#     # Calculating reference spectrum
#     xsp.Xset.chatter = 0
#     xsp.AllModels.setEnergies(str(emin) + " " + str(emax) + " " + str(len(energy)) + " lin")
#     if type(tbroad) == bool:
#         xsp.Xset.addModelString("APECTHERMAL", "yes" if tbroad else "no")
#     if abund:
#         xsp.Xset.abund = abund
#
#     pars = {}
#     pars[1] = temp
#     for ind in range(28):
#         pars[4 + ind] = metal
#     pars[32] = z
#     pars[33] = 1.
#     model = xsp.Model('vvapec', 'test_calc_spec', 0)
#     model.setPars(pars)
#     reference = np.array(model.values(0))
#
#     # Restoring PyXspec settings
#     xsp.Xset.chatter = chatter_
#     xsp.Xset.modelStrings = model_strings_
#     xsp.Xset.abund = abund_
#     xsp.AllModels.setEnergies("reset")  # Resets to the PyXspec default, not to the original
#
#     # Calculating spectrum from table
#     spec = calc_spec(specTable, z, temp, no_z_interp=True)
#
#     # Smoothing the arrays before checking the values as the lines may be smeared by the interpolation over temperaure
#     # def gaussian(x, mu, sig):
#     #    return 1./(np.sqrt(2.*np.pi)*sig)*np.exp(-np.square((x - mu)/sig)/2)
#     nsmooth = 50
#     smf = np.ones(nsmooth) / nsmooth
#     spec_sm = np.convolve(spec, smf, mode='valid')
#     reference_sm = np.convolve(reference, smf, mode='valid')
#
#     import matplotlib.pyplot as plt
#     plt.plot(spec_sm / reference_sm - 1)
#     assert all(val == pytest.approx(ref) for val, ref in zip(spec_sm, reference_sm)), (
#             "iz = " + str(iz) + ", temperature = " + str(temp) + ", seed = " + str(seed))
