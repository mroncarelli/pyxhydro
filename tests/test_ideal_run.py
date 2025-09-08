from xraysim.gadgetutils.convert import vpec2zobs
from xraysim.gadgetutils.phys_const import keV2K
from xraysim.sixte import cube2simputfile, create_eventlist, make_pha, versionTuple
from xraysim.sphprojection.mapping import make_speccube
from xraysim.specutils.specfit import *
from xraysim.specutils.tables import apec_table

from .specfittestutils import *

inputDir = os.environ.get('XRAYSIM') + '/tests/inp/'
snapshotFile = inputDir + 'snap_Gadget_sample'

# Emission table parameters
abund = "aspl"  # Xspec abundance table
ntemp = 40
tmin, tmax = 0.1, 50.  # Tempeature range [keV]
nene = 6000
e_min, e_max = 2., 8.  # Energy range [keV]
dz = 1e-3
nSigma = 5.  # Used to define the redshift range in units of the velocity dispersion defined below

# Map parameters
npix = 50

# Physical parameters
temp_keV = 5.  # Gas temperature [keV]
z = 0.2  # redshift [---]
metal = 0.35 # Metallicity
nH = 0.02  # Hydrogen column density [10^22 cm^-2]
sigma_v = 354.  # velocity dispersion [km/s]

# Calculating z_min and z_max for the spectral table: they correspond to z ± nSigma * sigma_v, rounded with dz
z_min = np.floor(vpec2zobs(-nSigma * sigma_v, z, 'km/s') / dz) * dz
z_max = np.ceil(vpec2zobs(nSigma * sigma_v, z, 'km/s') / dz) * dz
nz = int((z_max - z_min) / dz) + 1

referenceDir = os.environ.get('XRAYSIM') + '/tests/reference_files/'

spFile = referenceDir + "sp_file_created_for_ideal_test.simput"
simputFile = referenceDir + "simput_file_created_for_ideal_test.simput"
evtFile = referenceDir + "evt_file_created_for_ideal_test.evt"
phaFile = referenceDir + "pha_file_created_for_ideal_test.pha"


def test_ideal_run():
    """
    An ideal run from Gadget snapshot to spf file. No intermediate check is done, just checks that the final value of
    the physical parameters are correct.
    """

    # Creating the spectral table
    specTable = apec_table(nz, z_min, z_max, 2, temp_keV, temp_keV+1, nene, e_min, e_max, metal,
                           abund=abund)

    # Creating the speccube from the snapshot assuming isothermal gas with Gaussian velocity distribution
    specCube = make_speccube(snapshotFile, specTable, 1.05, npix, z, center=[2500., 2500.], proj='z', tcut=1.e6,
                             isothermal=temp_keV * keV2K, nh=nH, gaussvel=(0., sigma_v))

    del specTable

    # Creating the SIMPUT file
    if os.path.isfile(simputFile):
        os.remove(simputFile)
    cube2simputfile(specCube, simputFile)
    del specCube

    # Creating an event-list file from the SIMPUT file
    if os.path.isfile(evtFile):
        os.remove(evtFile)
    sys_out = create_eventlist(simputFile, 'xrism-resolve-test', 1.e6, evtFile, background=False,
                               seed=42, verbosity=0)
    assert sys_out == [0]
    os.remove(simputFile)

    # Creating a pha from the event-list file
    if os.path.isfile(phaFile):
        os.remove(phaFile)
    make_pha(evtFile, phaFile, grading=1) if versionTuple < (3,) else make_pha(evtFile, phaFile)
    os.remove(evtFile)
    assert os.path.isfile(phaFile)

    # Fitting the spectrum in the pha file with parameters starting with the right values
    specfitRightStart = SpecFit(phaFile, "phabs(bapec)")
    rightStartPars = (nH, # np.random.uniform(low=0., high=0.03),  # nH [10^22 cm^-2]
                      temp_keV,  # kT [keV]
                      metal,     # Abundance [Solar]
                      z,         # Redshift [---]
                      sigma_v,   # Velocity dispersion [km/s]
                      1.)        # Normalization (actually this is not right, but it works nonetheless)

    fixedPars = (True, False, False, False, False, False)
    specfitRightStart.run(start=rightStartPars, fixed=fixedPars, method='cstat', abund=abund, erange=(e_min, e_max))

    assert_fit_results_within_tolerance(specfitRightStart, (nH, temp_keV, metal, z, sigma_v,
                                                            specfitRightStart.fitResult["values"][5]), tol=2)
    del specfitRightStart

    # Fitting the spectrum in the pha file starting with wrong parameters
    specfitWrongStart = SpecFit(phaFile, "phabs(bapec)")
    startPars = (nH, # np.random.uniform(low=0., high=0.03),  # nH [10^22 cm^-2]
                 np.random.uniform(low=2., high=9.),    # kT [keV]
                 np.random.uniform(low=0., high=0.5),   # Abundance [Solar]
                 z,                                     # Redshift [---]
                 np.random.uniform(low=0., high=500.),  # Velocity dispersion [km/s]
                 1.)                                    # Normalization

    fixedPars = (True, False, False, False, False, False)
    specfitWrongStart.run(start=startPars, fixed=fixedPars, method='cstat', abund=abund, erange=(e_min, e_max))

    assert_fit_results_within_tolerance(specfitWrongStart, (nH, temp_keV, metal, z, sigma_v,
                                                            specfitWrongStart.fitResult["values"][5]), tol=2.5)
