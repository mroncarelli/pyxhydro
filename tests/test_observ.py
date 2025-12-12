from astropy.io import fits
import numpy as np
import os
import pytest

from xraysim.observ import countrate
from xraysim.sphprojection.mapping import read_speccube
from xraysim.gadgetutils.convert import ra_corr
from .randomutils import TrueRandomGenerator, globalRandomSeed

TRG = TrueRandomGenerator(globalRandomSeed)
errMsg = "Random seed: " + str(TRG.initialSeed)  # Assertion error message if test fails

inputDir = os.environ.get('XRAYSIM') + '/tests/inp/'
referenceDir = os.environ.get('XRAYSIM') + '/tests/reference_files/'
referenceSpcubeFile = referenceDir + 'reference.speccube'
referenceSimputFile = referenceDir + 'reference.simput'
referenceEvtFile = referenceDir + 'reference.evt'
evtTable = fits.open(referenceEvtFile)[1].data
evtTable['RA'] = ra_corr(evtTable['RA'], units='deg', zero=True)  # Correcting RA in the table events to zero-centered
instrumentFOV = 3  # [arcmin]
tExp = fits.open(referenceEvtFile)[1].header['EXPOSURE']  # [s]
sigmaTol = 3
arfFile = "/Users/mauro/Sixte/share/sixte/instruments/xrism-resolve-test/rsl_sixte_standard_GVclosed.arf"

# Getting minima and maxima of coordinates
spCube = read_speccube(referenceSpcubeFile)
emin = spCube["energy"][0] - 0.5 * spCube["energy_interval"][0]
emax = spCube["energy"][-1] + 0.5 * spCube["energy_interval"][-1]
size = spCube["size"] * 60  # [arcmin]
xmin, xmax = -0.5 * size, 0.5 * size  # [arcmin]
ymin, ymax = -0.5 * size, 0.5 * size  # [arcmin]

# Extracting random ranges
xrange = TRG.uniform(xmin, xmax, size=2)  # [arcmin]
xrange.sort()  # [arcmin]
yrange = TRG.uniform(ymin, ymax, size=2)  # [arcmin]
yrange.sort()  # [arcmin]
erange = TRG.uniform(emin, emax, size=2)  # [arcmin]
erange.sort()  # [arcmin]

# Ranges when considering the field-of-view
xrangeFOV = (-0.5 * instrumentFOV, 0.5 * instrumentFOV)  # [arcmin]
yrangeFOV = (-0.5 * instrumentFOV, 0.5 * instrumentFOV)  # [arcmin]

# Ranges when considering both
xrange2 = (max(xrangeFOV[0], xrange[0]), min(xrangeFOV[1], xrange[1]))  # [arcmin]
yrange2 = (max(yrangeFOV[0], yrange[0]), min(yrangeFOV[1], yrange[1]))  # [arcmin]


def test_countrate_of_speccube_must_be_the_same_with_different_input_type():
    """
    The countrate of a speccube calculated from the file or from the speccube dictionary must be identical.
    """
    assert (countrate(referenceSpcubeFile, arfFile) == countrate(read_speccube(referenceSpcubeFile), arfFile))


def test_countrate_of_simput_file_must_be_the_same_with_different_input_type():
    """
    The countrate of a Simput file calculated from the file or from the FITS HDUList must be identical.
    """
    assert countrate(referenceSimputFile, arfFile) == countrate(fits.open(referenceSimputFile), arfFile)


def test_countrate_of_speccube_and_simput_file_must_be_the_same():
    """
    The countrates of referenceSpcubeFile and of referenceSimputFile must be the same, as the latter was created
    starting from the former.
    """
    assert countrate(referenceSpcubeFile, arfFile) == pytest.approx(countrate(referenceSimputFile, arfFile))


def test_countrate_of_speccube_and_simput_file_with_xrange_must_be_the_same():
    """
    The countrates of referenceSpcubeFile and of referenceSimputFile with the same value of xrange must be the same,
    as the latter was created starting from the former.
    """
    assert (countrate(referenceSpcubeFile, arfFile, xrange=xrange) ==
            pytest.approx(countrate(referenceSimputFile, arfFile, xrange=xrange))), errMsg


def test_countrate_of_speccube_and_simput_file_with_yrange_must_be_the_same():
    """
    The countrates of referenceSpcubeFile and of referenceSimputFile with the same value of yrange must be the same,
    as the latter was created starting from the former.
    """
    assert (countrate(referenceSpcubeFile, arfFile, yrange=yrange) ==
            pytest.approx(countrate(referenceSimputFile, arfFile, yrange=yrange))), errMsg


def test_countrate_of_speccube_and_simput_file_with_erange_must_be_the_same():
    """
    The countrates of referenceSpcubeFile and of referenceSimputFile with the same value of erange must be the same,
    as the latter was created starting from the former.
    """
    assert (countrate(referenceSpcubeFile, arfFile, erange=erange) ==
            pytest.approx(countrate(referenceSimputFile, arfFile, erange=erange))), errMsg


def test_countrate_of_speccube_and_simput_file_with_xrange_yrange_erange_must_be_the_same():
    """
    The countrates of referenceSpcubeFile and of referenceSimputFile with the same value of xrange, yrange and erange
    must be the same, as the latter was created starting from the former.
    """
    assert (countrate(referenceSpcubeFile, arfFile, xrange=xrange, yrange=yrange, erange=erange) ==
            pytest.approx(countrate(referenceSimputFile, arfFile, xrange=xrange, yrange=yrange, erange=erange))), errMsg


def test_countrate_outside_xrange_must_be_zero():
    """
    The countrate computed outside the field in the x-coordinate must be zero.
    """
    assert countrate(referenceSpcubeFile, arfFile, xrange=(xmin - 2, xmin -1)) == 0
    assert countrate(referenceSimputFile, arfFile, xrange=(xmin - 2, xmin -1)) == 0


def test_countrate_outside_yrange_must_be_zero():
    """
    The countrate computed outside the field in the y-coordinate must be zero.
    """
    assert countrate(referenceSpcubeFile, arfFile, yrange=(ymin - 2, ymin -1)) == 0
    assert countrate(referenceSimputFile, arfFile, yrange=(ymin - 2, ymin -1)) == 0


def test_countrate_outside_erange_must_be_zero():
    """
    The countrate computed outside energy range must be zero.
    """
    assert countrate(referenceSpcubeFile, arfFile, erange=(emin - 2, emin -1)) == 0
    assert countrate(referenceSimputFile, arfFile, erange=(emin - 2, emin -1)) == 0


def nevt_filter(table: fits.fitsrec.FITS_rec, xrange=None, yrange=None, erange=None):
    """
    Counts the number of events given a filter.
    :param table: (fits.fitsrec.FITS_rec) Event-list table.
    :param xrange: (2 x float) Range in the x-axis, i.e. RA [arcmin]. Default None.
    :param yrange: (2 x float) Range in the y-axis, i.e. DEC [arcmin]. Default None.
    :param erange: (2 x float) Energy range [keV]. Default None.
    :return: (int) Number of events that match the filter.
    """

    if xrange is not None:
        table = table[np.where((table['RA'] * 60 >= xrange[0]) & (table['RA'] * 60 < xrange[1]))[0]]
    if yrange is not None:
        table = table[np.where((table['DEC'] * 60 >= yrange[0]) & (table['DEC'] * 60 < yrange[1]))[0]]
    if erange is not None:
        table = table[np.where((table['SIGNAL'] >= erange[0]) & (table['SIGNAL'] < erange[1]))[0]]

    return len(table)


def test_counts_in_eventlist_must_match_with_countrate():
    """
    The number of counts in the eventlist must match the ones calculated using the countrate and exposure.
    """
    ncts_from_ctrate = countrate(referenceSpcubeFile, arfFile, xrange=xrangeFOV, yrange=yrangeFOV) * tExp
    ncts_from_eventlist = nevt_filter(evtTable)

    assert ncts_from_ctrate == pytest.approx(ncts_from_eventlist, abs= sigmaTol * np.sqrt(ncts_from_eventlist))


def test_counts_in_eventlist_must_match_with_countrate_with_xrange():
    """
    The number of counts in the eventlist must match the ones calculated using the countrate and exposure with the same
    xrange.
    """
    ncts_from_ctrate = countrate(referenceSpcubeFile, arfFile, xrange=xrange2, yrange=yrangeFOV) * tExp
    ncts_from_eventlist = nevt_filter(evtTable, xrange=xrange)

    assert ncts_from_ctrate == pytest.approx(ncts_from_eventlist, abs= sigmaTol * np.sqrt(ncts_from_eventlist)), errMsg


def test_counts_in_eventlist_must_match_with_countrate_with_yrange():
    """
    The number of counts in the eventlist must match the ones calculated using the countrate and exposure with the same
    yrange.
    """
    ncts_from_ctrate = countrate(referenceSpcubeFile, arfFile, xrange=xrangeFOV, yrange=yrange2) * tExp
    ncts_from_eventlist = nevt_filter(evtTable, yrange=yrange)

    assert ncts_from_ctrate == pytest.approx(ncts_from_eventlist, abs= sigmaTol * np.sqrt(ncts_from_eventlist)), errMsg


def test_counts_in_eventlist_must_match_with_countrate_with_erange():
    """
    The number of counts in the eventlist must match the ones calculated using the countrate and exposure with the same
    energy range.
    """
    ncts_from_ctrate = countrate(referenceSpcubeFile, arfFile, xrange=xrangeFOV, yrange=yrangeFOV, erange=erange) * tExp
    ncts_from_eventlist = nevt_filter(evtTable, erange=erange)

    assert ncts_from_ctrate == pytest.approx(ncts_from_eventlist, abs= sigmaTol * np.sqrt(ncts_from_eventlist)), errMsg


def test_counts_in_eventlist_must_match_with_countrate_with_xrange_yrange_erange():
    """
    The number of counts in the eventlist must match the ones calculated using the countrate and exposure with the same
    xrange, yrange and energy range.
    """
    ncts_from_ctrate = countrate(referenceSpcubeFile, arfFile, xrange=xrange2, yrange=yrange2, erange=erange) * tExp
    ncts_from_eventlist = nevt_filter(evtTable, xrange=xrange, yrange=yrange, erange=erange)

    assert ncts_from_ctrate == pytest.approx(ncts_from_eventlist, abs= sigmaTol * np.sqrt(ncts_from_eventlist)), errMsg
