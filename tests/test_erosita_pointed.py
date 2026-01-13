import os
import warnings

import pytest
from astropy.io import fits

from pyxhydro.sixte import sixtesim, makespec, erosita_ccd_eventfile, instruments
from .fitstestutils import assert_hdu_list_matches_reference
from .__shared import (referenceDir, referenceErositaSimputFile, referenceErositaPointedEvtFile,
                       referenceErositaPointedPhaFile, clear_file)

evtFile = referenceDir + "evt_file_erosita_pointed_created_for_test.evt"
evtFile_ccdList = []
for ccd in range(1, 8):
    evtFile_ccdList.append(erosita_ccd_eventfile(evtFile, ccd))
phaFile = referenceDir + "pha_file_erosita_pointed_created_for_test.pha"

# Configuring skipping and warning
testInstrumentName = 'erosita-test'
testInstrument = instruments.get(testInstrumentName)
skipTest = testInstrument is None or not testInstrument.verify(verbose=0)
skipReason = "The '" + testInstrumentName + "' instrument is not present or not set up correctly."

# Introduced this option to address Issue #12. With the `standard` option the code does not test that the content of
# evtFile and phaFile match the reference as it may fail in some operative systems. With the `complete` option (pytest
# --eventlist complete) the contents are checked and the test fails if they don't match.
@pytest.fixture(scope="session")
def run_type(pytestconfig):
    return pytestconfig.getoption("eventlist").lower()


@pytest.mark.skipif(skipTest, reason=skipReason)
def test_erosita_pointed(run_type):
    """
    A run of a eROSITA pointed observation from SIMPUT file to pha file, checking that each intermediate step produces 
    a file compatible with reference one.
    """

    # Creating an event-list file from the SIMPUT file
    if os.path.isfile(evtFile):
        os.remove(evtFile)
    sys_out = sixtesim(referenceErositaSimputFile, testInstrumentName, 1.e4, evtFile,
                       background=False, seed=42, verbose=0)
    assert sys_out == [0, 0]

    # Removing CCD files
    for ccdFile in evtFile_ccdList:
        os.remove(ccdFile)

    if run_type == 'standard':
        # Checking only that the file was created
        assert os.path.isfile(evtFile)
        warnings.warn("Eventlist not checked. Run 'pytest --eventlist complete' to check it.")
    elif run_type == 'complete':
        # Checking that file content matches reference
        assert_hdu_list_matches_reference(fits.open(evtFile), fits.open(referenceErositaPointedEvtFile),
                                          key_skip=('DATE', 'CREADATE', 'COMMENT', 'CHECKSUM'),
                                          history_tag_skip=('START PARAMETER ', ' EvtFile = '))
    else:
        raise ValueError("ERROR in test_erosita_pointed.py: unknown option " + run_type)

    # Creating a pha from the event-list file
    if os.path.isfile(phaFile):
        os.remove(phaFile)
    makespec(referenceErositaPointedEvtFile, phaFile)
    os.remove(evtFile)

    if run_type == 'standard':
        # Checking only that the file was created
        assert os.path.isfile(phaFile)
        warnings.warn("Pha file not checked. Run 'pytest --eventlist complete' to check it.")
    elif run_type == 'complete':
        # Checking that file content matches reference
        assert_hdu_list_matches_reference(fits.open(phaFile), fits.open(referenceErositaPointedPhaFile),
                                          key_skip=('COMMENT'),
                                          history_tag_skip=('START PARAMETER ', ' Spectrum = '))
    else:
        raise ValueError("ERROR in test_erosita_pointed.py: unknown option " + run_type)

    os.remove(phaFile)


@pytest.fixture(scope="module", autouse=True)
def on_end_module():
    yield
    clear_file(evtFile)
    for ccd_ in range(1, 8):
        clear_file(erosita_ccd_eventfile(evtFile, ccd_))
    clear_file(phaFile)
