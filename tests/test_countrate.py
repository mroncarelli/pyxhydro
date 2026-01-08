from astropy.io import fits
import numpy as np
import pytest

from xraysim.observ import countrate, ra_corr
from xraysim.sphprojection.mapping import read_specmap
from xraysim import sixte
from .randomutils import TrueRandomGenerator, globalRandomSeed
from .__shared import referenceSpcubeFile, referenceSimputFile, referenceEvtFile

SP = np.float32

TRG = TrueRandomGenerator(globalRandomSeed)
errMsg = "Random seed: " + str(TRG.initialSeed)  # Assertion error message if test fails

evtTable = fits.open(referenceEvtFile)[1].data
evtTable['RA'] = ra_corr(evtTable['RA'], units='deg', zero=True)  # Correcting RA in the table events to zero-centered
instrumentFOV = 3  # [arcmin]
tExp = fits.open(referenceEvtFile)[1].header['EXPOSURE']  # [s]
sigmaTol = 3
instrumentName = 'xrism-resolve-test'
instrument = sixte.instruments.get(instrumentName)
arfFile = instrument.path + "/" + instrument.arf[0]

# Getting minima and maxima of coordinates
spCube = read_specmap(referenceSpcubeFile)
emin = spCube["energy"][0] - 0.5 * spCube["energy_interval"][0]
emax = spCube["energy"][-1] + 0.5 * spCube["energy_interval"][-1]
size = spCube["size"]  # [deg]
pixelSize = spCube["pixel_size"] / 60 # [deg]
xmin, xmax = -0.5 * size, 0.5 * size  # [deg]
ymin, ymax = -0.5 * size, 0.5 * size  # [deg]

# Extracting random ranges
xrange = TRG.uniform(xmin, xmax, size=2)  # [deg]
xrange.sort()  # [deg]
yrange = TRG.uniform(ymin, ymax, size=2)  # [deg]
yrange.sort()  # [deg]
erange = TRG.uniform(emin, emax, size=2)  # [deg]
erange.sort()  # [deg]

# For photon counts derived from the eventlist the range is reduced to avoid border effect. I also include a
# minimum range to avoid very small areas
xmin2, xmax2 = -0.5 * instrumentFOV + pixelSize, 0.5 * instrumentFOV - pixelSize # [deg]
ymin2, ymax2 = -0.5 * instrumentFOV + pixelSize, 0.5 * instrumentFOV - pixelSize  # [deg]
xrange2 = (0, 0)
while xrange2[1] - xrange2[0] < pixelSize:
    xrange2 = TRG.uniform(xmin2, xmax2, size=2)
yrange2 = (0, 0)
while yrange2[1] - yrange2[0] < pixelSize:
    yrange2 = TRG.uniform(ymin2, ymax2, size=2)


def test_countrate_of_speccube_must_be_the_same_with_different_input_type():
    """
    The countrate of a speccube calculated from the file or from the speccube dictionary must be identical.
    """
    assert (countrate(referenceSpcubeFile, arfFile) == countrate(read_specmap(referenceSpcubeFile), arfFile))


def test_countrate_must_be_the_same_with_different_arf_input_type():
    """
    The countrate of a speccube when the arf input is a file name, and HDUList, a sixte.Instrument or the instrument
    name must be identical.
    """
    ctrate_ref = countrate(referenceSpcubeFile, arfFile)
    assert countrate(referenceSpcubeFile, fits.open(arfFile)) == ctrate_ref
    assert countrate(referenceSpcubeFile, instrument) == ctrate_ref
    assert countrate(referenceSpcubeFile, instrumentName) == ctrate_ref


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


def nevt_filter(table: fits.fitsrec.FITS_rec, simput_file=None, xrange=None, yrange=None, erange=None):
    """
    Counts the number of events given a filter. For the position in the sky it is based on the coordinate of the source.
    :param table: (fits.fitsrec.FITS_rec) Event-list table.
    :param simput_file: (str) Simput file name, necessary only if xrange or yrange are present. Default None
    :param xrange: (2 x float) Range in the x-axis, i.e. RA [deg]. Default None.
    :param yrange: (2 x float) Range in the y-axis, i.e. DEC [deg]. Default None.
    :param erange: (2 x float) Energy range [keV]. Default None.
    :return: (int) Number of events that match the filter.
    """

    if xrange is not None or yrange is not None:
        src = fits.open(simput_file)[1].data
        if xrange is not None and yrange is None:
            ra = src['RA']  # [deg]
            src = src[np.where((ra >= xrange[0]) & (ra < xrange[1]))[0]]
            del ra
        elif xrange is None and yrange is not None:
            dec = src['DEC']  # [deg]
            src = src[np.where((dec >= yrange[0]) & (dec < yrange[1]))[0]]
            del dec
        elif xrange is not None and yrange is not None:
            ra = src['RA']  # [deg]
            dec = src['DEC']  # [deg]
            src = src[np.where((ra >= xrange[0]) & (ra < xrange[1]) & (dec >= yrange[0]) & (dec < yrange[1]))[0]]
            del ra, dec

        index_list = []
        for index in range(len(table)):
            if table['SRC_ID'][index] in src['SRC_ID']:
                index_list.append(index)

        table = table[index_list]

    if erange is not None:
        table = table[np.where((table['SIGNAL'] >= erange[0]) & (table['SIGNAL'] < erange[1]))[0]]

    return len(table)


def test_counts_in_eventlist_must_match_with_countrate():
    """
    The number of counts in the eventlist must match the ones calculated using the countrate and exposure.
    """
    ncts_from_ctrate = countrate(referenceSpcubeFile, arfFile) * tExp
    ncts_from_eventlist = nevt_filter(evtTable)

    assert ncts_from_ctrate == pytest.approx(ncts_from_eventlist, abs=sigmaTol * np.sqrt(ncts_from_eventlist))


def test_counts_in_eventlist_must_match_with_countrate_with_xrange():
    """
    The number of counts in the eventlist must match the ones calculated using the countrate and exposure with the same
    xrange.
    """
    ncts_from_ctrate = countrate(referenceSpcubeFile, arfFile, xrange=xrange2) * tExp
    ncts_from_eventlist = nevt_filter(evtTable, referenceSimputFile, xrange=xrange2)

    assert ncts_from_eventlist == pytest.approx(ncts_from_ctrate, abs=sigmaTol * np.sqrt(ncts_from_ctrate)), errMsg


def test_counts_in_eventlist_must_match_with_countrate_with_yrange():
    """
    The number of counts in the eventlist must match the ones calculated using the countrate and exposure with the same
    yrange.
    """
    ncts_from_ctrate = countrate(referenceSpcubeFile, arfFile, yrange=yrange2) * tExp
    ncts_from_eventlist = nevt_filter(evtTable, referenceSimputFile, yrange=yrange2)

    assert ncts_from_eventlist == pytest.approx(ncts_from_ctrate, abs=sigmaTol * np.sqrt(ncts_from_ctrate)), errMsg


def test_counts_in_eventlist_must_match_with_countrate_with_erange():
    """
    The number of counts in the eventlist must match the ones calculated using the countrate and exposure with the same
    energy range.
    """
    ncts_from_ctrate = countrate(referenceSpcubeFile, arfFile, erange=erange) * tExp
    ncts_from_eventlist = nevt_filter(evtTable, referenceSimputFile, erange=erange)

    assert ncts_from_eventlist == pytest.approx(ncts_from_ctrate, abs=sigmaTol * np.sqrt(ncts_from_ctrate)), errMsg


def test_counts_in_eventlist_must_match_with_countrate_with_xrange_yrange_erange():
    """
    The number of counts in the eventlist must match the ones calculated using the countrate and exposure with the same
    xrange, yrange and energy range.
    """
    ncts_from_ctrate = countrate(referenceSpcubeFile, arfFile, xrange=xrange2, yrange=yrange2, erange=erange) * tExp
    ncts_from_eventlist = nevt_filter(evtTable, referenceSimputFile, xrange=xrange2, yrange=yrange2, erange=erange)

    assert ncts_from_eventlist == pytest.approx(ncts_from_ctrate, abs=sigmaTol * np.sqrt(ncts_from_ctrate)), errMsg
