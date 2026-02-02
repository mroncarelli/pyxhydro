import pytest
import numpy as np
from astropy.io import fits

from pyxhydro.sphprojection.mapping import specmap, write_specmap
from pyxhydro.gadgetutils.phys_const import keV2K
from pyxhydro.specutils.tables import read_spectable, calc_spec

from .fitstestutils import assert_hdu_list_matches_reference
from .__shared import *

SP = np.float32

npix, size, redshift, center, proj, flag_ene, nsample, nh = 25, 0.05, 0.1, [2500., 2500.], 'z', False, 1, 0.01
nene = fits.open(referenceSpecTableFile)[0].header.get('NENE')
testFile = inputDir + 'file_created_for_test.spmap'

specMap = specmap(snapshotFile, referenceSpecTableFile, size=size, npix=npix, redshift=0.1, nh=nh,
                  center=center, proj=proj, tcut=1e6)


def test_structure(inp=specMap):
    """
    The output dictionary must contain all the keywords that must be present in every output
    :param inp: the spectral map dictionary to test
    """
    mandatory_keys_set = {'data', 'xrange', 'yrange', 'size', 'pixel_size', 'energy', 'energy_interval', 'units',
                          'coord_units', 'energy_units', 'simulation_file', 'spectral_table', 'proj', 'z_cos', 'd_c',
                          'flag_ene', 'smoothing', 'velocities'}
    for key in mandatory_keys_set:
        assert key in inp.keys()


def test_data_shape(inp=specMap):
    """
    The data in the output dictionary must be of the correct shape
    :param inp: the spectral map dictionary to test
    """
    assert inp.get('data').shape == (npix, npix, nene)


def test_key_values(inp=specMap):
    """
    Some data in the output dictionary must correspond exactly to the input ones
    :param inp: the spectral map dictionary to test
    """
    reference_dict = {'size': size, 'pixel_size': size / npix * 60., 'simulation_file': snapshotFile,
                      'spectral_table': referenceSpecTableFile, 'proj': proj, 'z_cos': redshift, 'flag_ene': flag_ene,
                      'smoothing': 'ON', 'velocities': 'ON'}

    for key in reference_dict:
        value = reference_dict.get(key)
        if type(value) == float:
            assert inp.get(key) == pytest.approx(value)
        else:
            assert inp.get(key) == value


def test_energy(inp=specMap):
    """
    The energy array in the dictionary must correspond to the one of the spectral table (if no ecut is present)
    :param inp: the spectral map dictionary to test
    """
    energy = inp.get('energy')
    energy_table = read_spectable(referenceSpecTableFile).get('energy')
    assert energy == pytest.approx(energy_table)


def test_isothermal_spectrum_with_temperature_from_table():
    """
    The spectrum of a specmap computed assuming isothermal gas with temperature taken directly from the table must
    have the same shape (i.e. non considering normalization) than the corresponding spectrum of the table.
    """
    sptable = read_spectable(referenceSpecTableFile)
    z_table = sptable.get('z')
    temperature_table = sptable.get('temperature')  # [keV]
    iz, it = len(z_table) // 2, len(temperature_table) // 2
    z, temp_iso = z_table[iz], temperature_table[it] * keV2K  # [K]
    spec_reference = sptable.get('data')[iz, it, :]
    spec_reference /= spec_reference.mean()  # normalize to mean = 1
    specmap_iso = specmap(snapshotFile, referenceSpecTableFile, size=size, npix=5, redshift=z, center=center, proj=proj,
                                  isothermal=temp_iso, novel=True, nsample=nsample).get('data')

    nene_specmap = specmap_iso.shape[2]
    spec_iso = np.ndarray(nene_specmap, dtype=SP)
    for iene in range(nene_specmap):
        spec_iso[iene] = specmap_iso[:, :, iene].sum()
    spec_iso /= spec_iso.mean()  # normalize to mean = 1

    assert spec_iso == pytest.approx(spec_reference, rel=1e-5)


def test_isothermal_spectrum():
    """
    The spectrum of a specmap computed assuming isothermal gas must have the same shape (i.e. non considering
    normalization) than the corresponding spectrum of the table.
    """
    sptable = read_spectable(referenceSpecTableFile)
    z_table = sptable.get('z')
    temperature_table = sptable.get('temperature')  # [keV]
    iz = len(z_table) // 2
    z = z_table[iz]
    temp_iso_kev = temperature_table[0] + 0.67 * (
                temperature_table[-1] - temperature_table[0])  # arbitrary value inside the table [keV]
    spec_reference = calc_spec(sptable, z, temp_iso_kev, no_z_interp=True)
    spec_reference /= spec_reference.mean()  # normalize to mean = 1
    temp_iso = temp_iso_kev * keV2K  # [K]
    specmap_iso = specmap(snapshotFile, referenceSpecTableFile, size=size, npix=5, redshift=z, center=center, proj=proj,
                                  isothermal=temp_iso, novel=True, nsample=nsample).get('data')

    spec_iso = np.ndarray(nene, dtype=SP)
    for iene in range(nene):
        spec_iso[iene] = specmap_iso[:, :, iene].sum()
    spec_iso /= spec_iso.mean()  # normalize to mean = 1

    assert spec_iso == pytest.approx(spec_reference, rel=1e-5)


def test_created_file_matches_reference(specmap_inp=specMap, reference=referenceSpmapFile):
    """
    Writing the specmap to a fits file should produce a file with data identical to the reference one.
    """
    if os.path.isfile(testFile):
        os.remove(testFile)
    write_specmap(specmap_inp, testFile)
    hdulist = fits.open(testFile)
    os.remove(testFile)
    hdulist_reference = fits.open(reference)

    assert_hdu_list_matches_reference(hdulist, hdulist_reference, tol=5e-5, warn_on_keys=True)


@pytest.fixture(scope="module", autouse=True)
def on_end_module():
    yield
    clear_file(testFile)
