from astropy import cosmology
from astropy.io import fits
import os
import numpy as np
import pygadgetreader as pygr
import pytest
import sys
sys.path.append(os.environ.get("HEADAS") + "/lib/python")
# TODO: the two lines above are necessary only to make the code work in IntelliJ (useful for debugging)
import xspec as xsp

from xraysim.gadgetutils.convert import vpec2zobs
from xraysim.gadgetutils.phys_const import keV2K, keV2erg, pi
from xraysim.sixte import cube2simputfile, create_eventlist, make_pha, versionTuple
from xraysim.sphprojection.mapping import make_map, make_speccube
from xraysim.specutils.specfit import SpecFit
from xraysim.specutils.tables import apec_table

from .randomutils import TrueRandomGenerator
from .specfittestutils import assert_fit_results_within_error

inputDir = os.environ.get('XRAYSIM') + '/tests/inp/'
snapshotFile = inputDir + 'snap_Gadget_sample'

# Emission table parameters
abund = "aspl"  # Xspec abundance table
nene = 6000
e_min, e_max = 2., 8.  # Energy range [keV]
d_ene = (e_max - e_min) / nene  # Energy bin [keV]
energy = np.linspace(e_min, e_max, num=nene, endpoint=False) + 0.5 * d_ene  # [keV]
dz = 1e-3
nSigmaVel = 5.  # Used to define the redshift range in units of the velocity dispersion defined below

# Map parameters
npix = 50

# Setting physical parameters of the tests with a true random generator
nHMin, nHMax = 0, 0.05
tMin, tMax = 2, 9
zMin, zMax = 0, 1.5
metalMin, metalMax = 0.2, 1
sigmaVMin, sigmaVMax = 50, 500

TRG = TrueRandomGenerator()
errMsg = "Random seed: " + str(TRG.initialSeed)  # Assertion error message if test fails
nH = TRG.uniform(nHMin, nHMax)  # Hydrogen column density [10^22 cm-2]
temp = TRG.uniform(tMin, tMax)  # Gas temperature [keV]
z = int(TRG.uniform(zMin, zMax) / dz) * dz  # redshift [---] (multiple of dz so that its values is in the table)
metal = TRG.uniform(metalMin, metalMax)  # metallicity [Solar]
sigma_v = TRG.uniform(sigmaVMin, sigmaVMax)  # velocity dispersion [km/s]

# Temperature ranges in the table
nt, tMinTable, tMaxTable = 3, temp / 1.5, temp * 1.5

# Calculating z_min and z_max for the spectral table: they correspond to z ± nSigma * sigma_v, rounded with dz
zMinTable = np.floor(vpec2zobs(-nSigmaVel * sigma_v, z, 'km/s') / dz) * dz
zMaxTable = np.ceil(vpec2zobs(nSigmaVel * sigma_v, z, 'km/s') / dz) * dz
nz = int(np.round((zMaxTable - zMinTable) / dz)) + 1
if nz < 3:
    zMinTable -= dz
    zMaxTable += dz
    nz = int(np.round((zMaxTable - zMinTable) / dz)) + 1

# Creating the spectral table
specTable = apec_table(nz, zMinTable, zMaxTable, nt, tMinTable, tMaxTable, nene, e_min, e_max, metal, abund=abund)

# Calculating normalization
cosmo = cosmology.FlatLambdaCDM(H0=100., Om0=0.3)
gadget2arcmin = cosmo.arcsec_per_kpc_comoving(z).to_value() / 60.  # 1 arcmin / 1 h^-1 kpc (comoving)
d_C = 1e3 * cosmo.comoving_distance(z).to_value()  # [h^-1 kpc] comoving
XRISM_FOV = 3.  # [arcmin]
mapSize = XRISM_FOV / gadget2arcmin # [h^-1 kpc] (comoving)
h_Hubble = pygr.readhead(snapshotFile, 'hubble')
map_str = make_map(snapshotFile, 'nenH', 1, center=[2500., 2500.], size=mapSize, struct=True, tcut=1e6)
InenHdl = map_str['map'][0, 0]  # [h^3 cn^-5] (comoving)
norm = InenHdl * 1e-14 * h_Hubble ** 3 * (1 + z) ** 3 * mapSize ** 2 / (4 * pi * d_C ** 2) # [10^14 cm^-5] (physical)

rightParsV0 = (nH, temp, metal, z, 0, norm)      # Normalization [10^14 cm^-5]
rightPars = (nH, temp, metal, z, sigma_v, norm)      # Normalization [10^14 cm^-5]

# Test files
referenceDir = os.environ.get('XRAYSIM') + '/tests/reference_files/'
spFile = referenceDir + "sp_file_created_for_ideal_test.simput"
simputFile = referenceDir + "simput_file_created_for_ideal_test.simput"
evtFile = referenceDir + "evt_file_created_for_ideal_test.evt"
phaFile = referenceDir + "pha_file_created_for_ideal_test.pha"


def wabs_bapec(nh: float, temp: float, metal: float, z: float, sigma_v: float, norm: float) -> np.ndarray:
    """
    Returns an absorbed (wabs) bapec spectrum.
    """
    xsp.Xset.chatter = 0
    xsp.Xset.addModelString("APECTHERMAL", "yes")
    xsp.Xset.abund = abund
    xsp.AllModels.setEnergies(str(e_min) + " " + str(e_max) + " " + str(nene) + " lin")

    pars = {}
    pars[1] = nH  # [10^22 cm^-2]
    pars[2] = temp  # [keV]
    for ind in range(28):
        pars[5 + ind] = metal
    pars[33] = z
    pars[34] = sigma_v
    pars[35] = norm
    model = xsp.Model('wabs(bvvapec)', 'test_ideal_run', 0)
    model.setPars(pars)
    result = np.array(model.values(0))  # [photons s^-1 cm^-2] (already multiplied by norm)
    xsp.AllModels.setEnergies("reset")

    return result


def test_isothermal_no_velocities():
    """
    An ideal run from Gadget snapshot, assuming isothermal gas with no velocities.
    """

    # Computing reference spectrum
    specRef = wabs_bapec(nH, temp, metal, z, 0, norm)  # [photons s^-1 cm^-2] (already multiplied by norm)


    # Creating the speccube from the snapshot assuming isothermal gas with no velocities velocity distribution
    specCube = make_speccube(snapshotFile, specTable, XRISM_FOV / 60., npix, z, center=[2500., 2500.], proj='z',
                             tcut=1e6, isothermal=temp * keV2K, nh=nH, novel=True)

    # Checking that the integrated spectrum matches with the reference one
    assert specCube['energy'] == pytest.approx(energy, rel=1e-6), errMsg  # [keV]
    specSpecCube = specCube['data'].sum(axis=(0, 1)) * d_ene * specCube['pixel_size'] ** 2  # [photons s^-1 cm^2]
    assert specSpecCube == pytest.approx(specRef, rel=1e-3), errMsg  # [photons s^-1 cm^2]

    # Creating the SIMPUT file
    if os.path.isfile(simputFile):
        os.remove(simputFile)
    cube2simputfile(specCube, simputFile)
    del specCube

    # Extracting data from Simput file
    hduList = fits.open(simputFile)
    flux = hduList[1].data['FLUX']  # [erg/s/cm**2]
    fld = hduList[2].data['FLUXDENSITY']  # [photons/s/cm**2/keV]
    assert fld.shape[0] == npix**2, errMsg
    assert fld.shape[1] == nene, errMsg
    specSimput = np.zeros(nene)
    for ind in range(npix**2):
        assert hduList[2].data['ENERGY'][ind, :] == pytest.approx(energy, rel=1e-6), errMsg
        flux_ = np.sum(fld[ind, :] * energy) * d_ene * keV2erg  # [erg s^-1 cm^-2]
        specSimput += flux[ind] / flux_ * fld[ind] * d_ene  # [photons s^-1 cm^-2]

    assert specSimput == pytest.approx(specRef, rel=1e-3), errMsg

    # Creating an event-list file from the SIMPUT file
    if os.path.isfile(evtFile):
        os.remove(evtFile)
    sys_out = create_eventlist(simputFile, 'xrism-resolve-test', 1.e5, evtFile, background=False,
                               seed=42, verbosity=0)
    assert sys_out == [0], errMsg
    os.remove(simputFile)

    # Creating a pha from the event-list file
    if os.path.isfile(phaFile):
        os.remove(phaFile)
    make_pha(evtFile, phaFile, grading=1) if versionTuple < (3,) else make_pha(evtFile, phaFile)
    os.remove(evtFile)
    assert os.path.isfile(phaFile), errMsg

    # Fitting the spectrum in the pha file with parameters starting with the right values
    specfitRightStart = SpecFit(phaFile, "wabs(bapec)")

    fixedPars = (True, False, False, False, True, False)
    specfitRightStart.run(start=rightParsV0, fixed=fixedPars, method='cstat', abund=abund, erange=(e_min, e_max))

    assert_fit_results_within_error(specfitRightStart, rightParsV0, tol=1.5, msg=errMsg)
    del specfitRightStart

    # Fitting the spectrum in the pha file starting with wrong parameters
    specfitWrongStart = SpecFit(phaFile, "wabs(bapec)")
    startPars = (nH,  # nH [10^22 cm^-2]
                 TRG.uniform(low=tMin, high=tMax),  # kT [keV]
                 TRG.uniform(low=metalMin, high=metalMax),  # Abundance [Solar]
                 z,  # Redshift [---]
                 0,  # Velocity dispersion [km/s]
                 1)                                    # Normalization [10^14 cm^-5]

    fixedPars = (True, False, False, False, True, False)
    specfitWrongStart.run(start=startPars, fixed=fixedPars, method='cstat', abund=abund, erange=(e_min, e_max))

    assert_fit_results_within_error(specfitWrongStart, rightParsV0, tol=4, msg=errMsg)
