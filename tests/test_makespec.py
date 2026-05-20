import os
import pytest

from pyxhydro.sixte import instruments, makespec, get_filter_values

from .__shared import testInstrumentName, referenceDir, referenceEvtFile, clear_file

testInstrument = instruments.get(testInstrumentName)

phaFile = referenceDir + "pha_file_created_for_test_makespec.pha"
gradingList = [4, 2, -38, -37, -36, 0, 1, -16]  # Includes also non-realistic negative values
pixIdList = [113, 41, 42, 9, 56, 57, 58, 1007, 618, 616, 617]


def test_pha_file_created():
    """
    The pha file must be created correctly.
    """
    if os.path.isfile(phaFile):
        os.remove(phaFile)
    makespec(referenceEvtFile, phaFile, rsppath=testInstrument.path)
    assert os.path.isfile(phaFile)
    os.remove(phaFile)


@pytest.mark.filterwarnings("ignore")
def test_pixid_and_grading_empty_on_pha_without_filter():
    """
    Calling get_filter_values on a pha file with no filter must return an empty integer
    array.
    """
    if os.path.isfile(phaFile):
        os.remove(phaFile)
    makespec(referenceEvtFile, phaFile, rsppath=testInstrument.path)
    assert os.path.isfile(phaFile)
    assert len(get_filter_values(phaFile)) == 0
    assert len(get_filter_values(phaFile, key='GRADING')) == 0
    os.remove(phaFile)


@pytest.mark.filterwarnings("ignore")
def test_get_filter_values_recovers_pixid():
    """
    The PIXID values must be recovered correctly from the pha file. It also checks that
    with GRADING you get an empty integer array.
    """
    if os.path.isfile(phaFile):
        os.remove(phaFile)
    makespec(referenceEvtFile, phaFile, rsppath=testInstrument.path, pixid=pixIdList)
    assert os.path.isfile(phaFile)
    assert len(get_filter_values(phaFile, key='GRADING')) == 0
    pix_id = get_filter_values(phaFile)  # key has PIXID as default
    os.remove(phaFile)
    assert set(pix_id) == set(pixIdList)


@pytest.mark.filterwarnings("ignore")
def test_get_filter_values_recovers_grading():
    """
    The GRADING values must be recovered correctly from the pha file. It also checks that
    with PIXID you get an empty integer array.
    """
    if os.path.isfile(phaFile):
        os.remove(phaFile)
    makespec(referenceEvtFile, phaFile, rsppath=testInstrument.path, grading=gradingList)
    assert os.path.isfile(phaFile)
    assert len(get_filter_values(phaFile, key='PIXID')) == 0
    grading = get_filter_values(phaFile, key='GRADING')
    os.remove(phaFile)
    assert set(grading) == set(gradingList)


def test_get_filter_values_recovers_pixid_and_grading():
    """
    The GRADING values must be recovered correctly from the pha file.
    """
    if os.path.isfile(phaFile):
        os.remove(phaFile)
    makespec(referenceEvtFile, phaFile, rsppath=testInstrument.path, pixid=pixIdList, grading=gradingList)
    assert os.path.isfile(phaFile)
    pix_id = get_filter_values(phaFile)  # key has PIXID as default
    assert os.path.isfile(set(pix_id) == set(pixIdList))
    grading = get_filter_values(phaFile, key='GRADING')
    os.remove(phaFile)
    assert set(grading) == set(gradingList)


@pytest.fixture(scope="module", autouse=True)
def on_end_module():
    yield
    clear_file(phaFile)
