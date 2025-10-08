from astropy import cosmology
import pygadgetreader as pygr
import pytest

from xraysim.gadgetutils.convert import vpec2zobs
from xraysim.gadgetutils.phys_const import keV2K, keV2erg, pi
from xraysim.sixte import cube2simputfile # , create_eventlist, make_pha, versionTuple
from xraysim.sphprojection.mapping import make_map, make_speccube
from xraysim.specutils.specfit import *
from xraysim.specutils.tables import apec_table

inputDir = os.environ.get('XRAYSIM') + '/tests/inp/'
snapshotFile = inputDir + 'snap_Gadget_sample'

# Emission table parameters
abund = "aspl"  # Xspec abundance table
nH = 0.02  # [10^22 cm-2]
tmin, tmax = 0.1, 50.  # Tempeature range [keV]
nene = 6000
e_min, e_max = 2., 8.  # Energy range [keV]
d_ene = (e_max - e_min) / nene  # Energy bin [keV]
energy = np.linspace(e_min, e_max, num=nene, endpoint=False) + 0.5 * d_ene  # [keV]
dz = 1e-3
nSigma = 5.  # Used to define the redshift range in units of the velocity dispersion defined below

# Map parameters
npix = 50

# Physical parameters
temp_keV = 5.  # Gas temperature [keV]
z = 0.5  # redshift [---]
metal = 0.35 # Metallicity
sigma_v = 350.  # velocity dispersion [km/s]

# Calculating z_min and z_max for the spectral table: they correspond to z ± nSigma * sigma_v, rounded with dz
z_min = np.floor(vpec2zobs(-nSigma * sigma_v, z, 'km/s') / dz) * dz
z_max = np.ceil(vpec2zobs(nSigma * sigma_v, z, 'km/s') / dz) * dz
nz = int((z_max - z_min) / dz) + 1
if nz < 3:
    z_min -= dz
    z_max += dz
    nz = int((z_max - z_min) / dz) + 1

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

# Test files
referenceDir = os.environ.get('XRAYSIM') + '/tests/reference_files/'
spFile = referenceDir + "sp_file_created_for_ideal_test.simput"
simputFile = referenceDir + "simput_file_created_for_ideal_test.simput"
evtFile = referenceDir + "evt_file_created_for_ideal_test.evt"
phaFile = referenceDir + "pha_file_created_for_ideal_test.pha"


# TODO: This is a draft version of the test. The last commented part fails as there seems to be something wrong with
# the normalization (see Issue #23)
def test_isothermal_no_velocities():
    """
    An ideal run from Gadget snapshot, assuming isothermal gas with no velocites.
    """

    # Computing reference spectrum
    xsp.Xset.chatter = 0
    xsp.Xset.addModelString("APECTHERMAL", "yes")
    xsp.Xset.abund = abund
    xsp.AllModels.setEnergies(str(e_min) + " " + str(e_max) + " " + str(nene) + " lin")

    pars = {}
    pars[1] = nH  # [10^22 cm^-2]
    pars[2] = temp_keV  # [keV]
    for ind in range(28):
        pars[5 + ind] = metal
    pars[33] = z
    pars[34] = 1
    model = xsp.Model('wabs(vvapec)', 'test_ideal_run_isoth_novel', 0)
    model.setPars(pars)
    specRef = norm * np.array(model.values(0))  # [photons s^-1 cm^-2] (already multiplied by norm)
    xsp.AllModels.setEnergies("reset")

    # Creating the spectral table
    specTable = apec_table(nz, z_min, z_max, 2, temp_keV-0.5, temp_keV, nene, e_min, e_max, metal, abund=abund)

    # Creating the speccube from the snapshot assuming isothermal gas with Gaussian velocity distribution
    specCube = make_speccube(snapshotFile, specTable, XRISM_FOV/60., npix, z, center=[2500., 2500.], proj='z', tcut=1e6,
                             isothermal=temp_keV * keV2K, nh=nH, novel=True)

    # del specTable

    # Checking that the integrated spectrum matches with the reference one
    assert specCube['energy'] == pytest.approx(energy, rel=1e-6)  # [keV]
    specSpecCube = specCube['data'].sum(axis=(0, 1)) * d_ene * specCube['pixel_size'] ** 2  # [photons s^-1 cm^2]
    assert specSpecCube == pytest.approx(specRef, rel=1e-3)  # [photons s^-1 cm^2]

    # Creating the SIMPUT file
    if os.path.isfile(simputFile):
        os.remove(simputFile)
    cube2simputfile(specCube, simputFile)
    del specCube

    # Extracting data from Simput file
    hduList = fits.open(simputFile)
    os.remove(simputFile)
    flux = hduList[1].data['FLUX']  # [erg/s/cm**2]
    fld = hduList[2].data['FLUXDENSITY']  # [photons/s/cm**2/keV]
    assert fld.shape[0] == npix**2
    assert fld.shape[1] == nene
    specSimput = np.zeros(nene)
    for ind in range(npix**2):
        assert hduList[2].data['ENERGY'][ind, :] == pytest.approx(energy, rel=1e-6)
        flux_ = np.sum(fld[ind, :] * energy) * d_ene * keV2erg  # [erg s^-1 cm^-2]
        specSimput += flux[ind] / flux_ * fld[ind] * d_ene  # [photons s^-1 cm^-2]

    assert specSimput == pytest.approx(specRef, rel=1e-3)

    # Creating an event-list file from the SIMPUT file TODO decomment after solving Issue #23 (imports needed)
    # if os.path.isfile(evtFile):
    #     os.remove(evtFile)
    # sys_out = create_eventlist(simputFile, 'xrism-resolve-test', 1.e5, evtFile, background=False,
    #                            seed=42, verbosity=0)
    # assert sys_out == [0]
    # os.remove(simputFile)
    #
    # # Creating a pha from the event-list file
    # if os.path.isfile(phaFile):
    #     os.remove(phaFile)
    # make_pha(evtFile, phaFile, grading=1) if versionTuple < (3,) else make_pha(evtFile, phaFile)
    # os.remove(evtFile)
    # assert os.path.isfile(phaFile)
    #
    # # Fitting the spectrum in the pha file with parameters starting with the right values
    # specfitRightStart = SpecFit(phaFile, "phabs(bapec)")
    # rightStartPars = (nH, # np.random.uniform(low=0., high=0.03),  # nH [10^22 cm^-2]
    #                   temp_keV,  # kT [keV]
    #                   metal,     # Abundance [Solar]
    #                   z,         # Redshift [---]
    #                   sigma_v,   # Velocity dispersion [km/s]
    #                   norm)      # Normalization [10^14 cm^-5]
    #
    # fixedPars = (True, True, True, True, True, False)
    # specfitRightStart.run(start=rightStartPars, fixed=fixedPars, method='cstat', abund=abund, erange=(e_min, e_max))
    #
    # assert_fit_results_within_tolerance(specfitRightStart, (nH, temp_keV, metal, z, sigma_v,
    #                                                         norm), tol=3)
    # del specfitRightStart
    #
    # # Fitting the spectrum in the pha file starting with wrong parameters
    # specfitWrongStart = SpecFit(phaFile, "phabs(bapec)")
    # startPars = (nH, # np.random.uniform(low=0., high=0.03),  # nH [10^22 cm^-2]
    #              np.random.uniform(low=2., high=9.),    # kT [keV]
    #              np.random.uniform(low=0., high=0.5),   # Abundance [Solar]
    #              z,                                     # Redshift [---]
    #              np.random.uniform(low=0., high=500.),  # Velocity dispersion [km/s]
    #              1.)                                    # Normalization [10^14 cm^-5]
    #
    # fixedPars = (True, False, False, False, False, False)
    # specfitWrongStart.run(start=startPars, fixed=fixedPars, method='cstat', abund=abund, erange=(e_min, e_max))
    #
    # assert_fit_results_within_tolerance(specfitWrongStart, (nH, temp_keV, metal, z, sigma_v,
    #                                                         norm), tol=3.5)
