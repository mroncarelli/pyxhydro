import warnings
import pytest

from xraysim.sixte import cube2simputfile, create_eventlist, make_pha, versionTuple
from xraysim.sphprojection.mapping import make_speccube, write_speccube, read_speccube
from xraysim.specutils.specfit import *
from .fitstestutils import assert_hdu_list_matches_reference
from .specfittestutils import assert_specfit_has_coherent_properties

inputDir = os.environ.get('XRAYSIM') + '/tests/inp/'
referenceDir = os.environ.get('XRAYSIM') + '/tests/reference_files/'
snapshotFile = inputDir + 'snap_Gadget_sample'
spFile = referenceDir + 'reference_emission_table.fits'
referenceSpcubeFile = referenceDir + 'reference.speccube'
referenceSimputFile = referenceDir + 'reference.simput'
referenceEvtFile = referenceDir + 'reference.evt'
referencePhaFile = referenceDir + 'reference.pha'
referenceSpfFile = referenceDir + 'reference.spf'

spcubeFile = referenceDir + "spcube_file_created_for_test.spcube"
spcubeFile2 = referenceDir + "spcube_file_created_for_test_2.spcube"
simputFile = referenceDir + "simput_file_created_for_test.simput"
evtFile = referenceDir + "evt_file_created_for_test.evt"
phaFile = referenceDir + "pha_file_created_for_test.pha"
spfFile = referenceDir + "spf_file_created_for_test.spf"

# Introduced this option to address Issue #12. With the `standard` option the code does not test that the content of
# evtFile and phaFile match the reference as it may fail in some operative systems. With the `complete` option (pytest
# --eventlist complete) the contents are checked and the test fails if they don't match.
@pytest.fixture(scope="session")
def run_type(pytestconfig):
    return pytestconfig.getoption("eventlist").lower()


@pytest.mark.filterwarnings("ignore")
def test_full_run(run_type):
    """
    A full run from Gadget snapshot to spf file, checking that each intermediate step produces a file compatible with
    reference one.
    """

    # Creating a speccube file from a calculated speccube
    speccube_calculated = make_speccube(snapshotFile, spFile, 1.05, 25, redshift=0.1, center=[2500., 2500.],
                                        proj='z', tcut=1.e6, nh=0.01, nsample=1)
    if os.path.isfile(spcubeFile):
        os.remove(spcubeFile)
    write_speccube(speccube_calculated, spcubeFile)
    assert os.path.isfile(spcubeFile)
    del speccube_calculated

    reference_speccube = fits.open(referenceSpcubeFile)

    # Checking that file content matches reference
    assert_hdu_list_matches_reference(fits.open(spcubeFile), reference_speccube, tol=5e-5)

    # Creating a speccube file from the speccube read from the file
    speccube_read = read_speccube(spcubeFile)
    os.remove(spcubeFile)
    if os.path.isfile(spcubeFile2):
        os.remove(spcubeFile2)
    write_speccube(speccube_read, spcubeFile2)
    assert os.path.isfile(spcubeFile2)

    # Checking that file content matches reference
    assert_hdu_list_matches_reference(fits.open(spcubeFile2), reference_speccube, tol=5e-5)
    os.remove(spcubeFile2)

    # Creating a SIMPUT file from a speccube
    if os.path.isfile(simputFile):
        os.remove(simputFile)
    cube2simputfile(speccube_read, simputFile)
    del speccube_read

    # Checking that file content matches reference
    assert_hdu_list_matches_reference(fits.open(simputFile), fits.open(referenceSimputFile), tol=5e-5)

    # Creating an event-list file from the SIMPUT file
    if os.path.isfile(evtFile):
        os.remove(evtFile)
    sys_out = create_eventlist(referenceSimputFile, 'xrism-resolve-test', 1.e5, evtFile,
                               background=False, seed=42, verbosity=0)
    assert sys_out == [0]
    os.remove(simputFile)

    if run_type == 'standard':
        # Checking only that the file was created
        assert os.path.isfile(evtFile)
        warnings.warn("Eventlist not checked. Run 'pytest --eventlist complete' to check it.")
    elif run_type == 'complete':
        # Checking that file content matches reference
        assert_hdu_list_matches_reference(fits.open(evtFile), fits.open(referenceEvtFile), tol=5e-5,
                                          key_skip=('DATE', 'CREADATE', 'COMMENT'),
                                          history_tag_skip=('START PARAMETER ', ' EvtFile = '))
    else:
        raise ValueError("ERROR in test_full_run.py: unknown option " + run_type)

    # Creating a pha from the event-list file
    if os.path.isfile(phaFile):
        os.remove(phaFile)
    make_pha(referenceEvtFile, phaFile, grading=1) if versionTuple < (3,) else make_pha(referenceEvtFile, phaFile)
    os.remove(evtFile)
    assert os.path.isfile(phaFile)

    if run_type == 'standard':
        warnings.warn("Pha file not checked. Run 'pytest --eventlist complete' to check it.")
    elif run_type == 'complete':
        # Checking that file content matches reference
        assert_hdu_list_matches_reference(fits.open(phaFile), fits.open(referencePhaFile), tol=5e-5,
                                          key_skip=('COMMENT'),
                                          history_tag_skip=('START PARAMETER ', ' Spectrum = '))
    else:
        raise ValueError("ERROR in test_full_run.py: unknown option " + run_type)
    os.remove(phaFile)

    # Fitting the spectrum in the pha file
    specfit = SpecFit(referencePhaFile, "wabs(bapec)")
    assert_specfit_has_coherent_properties(specfit)
    startPars = (0.01, 6., 0.3, 0.1, 1000., 1e-3)
    fixedPars = (True, False, True, False, False, False)
    specfit.run(start=startPars, fixed=fixedPars, method='cstat', abund='aspl', erange=(3, 7))
    assert_specfit_has_coherent_properties(specfit)
    specfit.save(spfFile, overwrite=True)
    assert os.path.isfile(spfFile)
    assert_hdu_list_matches_reference(fits.open(spfFile), fits.open(referenceSpfFile), tol=1e-4)
    os.remove(spfFile)
