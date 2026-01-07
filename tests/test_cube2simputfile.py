import os

import pytest
from astropy.io import fits

from xraysim.gadgetutils.phys_const import keV2K
from xraysim.sixte import cube2simputfile
from xraysim.specutils.tables import read_spectable, calc_spec
from xraysim.sphprojection.mapping import make_speccube
from .fitstestutils import assert_hdu_list_matches_reference
from .__shared import inputDir, referenceSpecTableFile, referenceSimputFile, snapshotFile, clear_file

environmentVariablesPathList = [os.environ.get('XRAYSIM'), os.environ.get('SIXTE')]
npix, size, redshift, proj, flag_ene, tcut, nsample, nh = 25, 0.05, 0.1, 'z', False, 1.e6, 1, 0.01
t_iso_keV = 6.3  # [keV]
t_iso = t_iso_keV * keV2K  # [K]
center = [2500., 2500.]  # comoving [h^-1 kpc]
nene = fits.open(referenceSpecTableFile)[0].header.get('NENE')
testSimputFile = inputDir + 'file_created_for_test.simput'

# Isothermal + no velocities
speccubeIsothermalNovel = make_speccube(snapshotFile, referenceSpecTableFile, size=size, npix=npix, redshift=redshift,
                                        center=center, proj=proj, nsample=nsample, isothermal=t_iso, novel=True)

speccube = make_speccube(snapshotFile, referenceSpecTableFile, size=size, npix=npix, redshift=redshift, center=center,
                         proj=proj, tcut=tcut, nh=nh, nsample=nsample)


def test_file_created(inp=speccubeIsothermalNovel, out=testSimputFile):
    """
    The output SIMPUT file must be correctly created
    :param inp: (dict) spectral cube structure
    :param out: (str) output SIMPUT file
    """
    if os.path.isfile(out):
        os.remove(out)
    cube2simputfile(inp, out)
    assert os.path.isfile(out)
    os.remove(out)


def test_primary_header_keywords(inp=speccubeIsothermalNovel, out=testSimputFile):
    """
    The header of the Primary of the output SIMPUT file must contain a series of keywords whose value depend on the
    input data
    :param inp: (dict) spectral cube structure
    :param out: (str) output SIMPUT file
    """
    if os.path.isfile(out):
        os.remove(out)
    center_map = (12., 34.)
    cube2simputfile(inp, out, pos=center_map)
    header = fits.open(out)[0].header
    os.remove(out)
    assert header.get('SIM_FILE') == inp.get('simulation_file')
    assert header.get('SP_FILE') == inp.get('spectral_table')
    assert header.get('PROJ') == inp.get('proj')
    assert header.get('Z_COS') == pytest.approx(inp.get('z_cos'))
    assert (header.get('D_C'), header.comments['D_C']) == pytest.approx((inp.get('d_c'), '[h^-1 kpc]'))
    assert header.get('NPIX') == inp.get('data').shape[0]
    assert header.get('NENE') == inp.get('data').shape[2]
    assert (header.get('ANG_PIX'), header.comments['ANG_PIX']) == pytest.approx((inp.get('pixel_size'), '[arcmin]'))
    assert (header.get('ANG_MAP'), header.comments['ANG_MAP']) == pytest.approx((inp.get('size'), '[deg]'))
    assert (header.get('RA_C'), header.comments['RA_C']) == pytest.approx((center_map[0], '[deg]'))
    assert (header.get('DEC_C'), header.comments['DEC_C']) == pytest.approx((center_map[1], '[deg]'))
    assert header.get('SMOOTH') == inp.get('smoothing')
    assert header.get('VPEC') == inp.get('velocities')


def test_isothermal_spectrum(inp=speccubeIsothermalNovel, out=testSimputFile):
    """
    All the spectra contained in the Extension 2 of the SIMPUT file must be isothermal with T equal to the value
    indicated by the spec_cube, with arbitrary normalization.
    :param inp: (dict) spectral cube structure
    :param out: (str) output SIMPUT file
    """

    # Creating SIMPUT file
    if os.path.isfile(out):
        os.remove(out)
    cube2simputfile(inp, out)
    hdulist = fits.open(testSimputFile)
    os.remove(out)
    header0 = hdulist[0].header

    # Reading the spectral table and the temperature from the header of the SIMPUT file just created
    sptable = read_spectable(header0.get('SP_FILE'))
    temp = header0.get('ISO_T') / keV2K  # [keV]

    # Getting energy and (normalized spectrum) from table and from
    z = speccubeIsothermalNovel.get('z_cos')
    energy_reference = speccubeIsothermalNovel.get('energy')  # [keV]
    spectrum_reference = calc_spec(sptable, z, temp, no_z_interp=True)
    spectrum_reference /= spectrum_reference.mean()  # normalize to mean = 1

    # Checking energy from the SIMPUT file
    for energy in hdulist[2].data['ENERGY']:
        assert energy == pytest.approx(energy_reference, rel=1e-6)

    # Checking spectra from the SIMPUT file
    for spectrum in hdulist[2].data['FLUXDENSITY']:
        spectrum_norm = spectrum / spectrum.mean()  # normalize to mean = 1
        assert spectrum_norm == pytest.approx(spectrum_reference, rel=1e-6)


def test_created_file_matches_reference(inp=speccube, out=testSimputFile, reference=referenceSimputFile):
    """
    Writing the spec_cube to a SIMPUT file should produce a file with data identical to the reference one.
    """
    if os.path.isfile(testSimputFile):
        os.remove(testSimputFile)
    cube2simputfile(inp, out)
    hdulist = fits.open(testSimputFile)
    os.remove(out)
    hdulist_reference = fits.open(reference)
    assert_hdu_list_matches_reference(hdulist, hdulist_reference, tol=5e-5)


@pytest.fixture(scope="module", autouse=True)
def on_end_module():
    yield
    clear_file(testSimputFile)
