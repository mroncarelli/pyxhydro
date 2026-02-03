import pytest
from astropy.io import fits
from pyxhydro.specutils.tables import apec_table, write_spectable

from .fitstestutils import assert_hdu_list_matches_reference
from .__shared import referenceSpecTableFile, inputDir, clear_file

specTableFile = inputDir + '/emission_table_created_for_test.fits'


def test_create_apec_table_matches_reference():
    """
    The table created with the apec_table procedure should match the reference one.
    """
    nz, zmin, zmax = 6, 0.07, 0.13
    ntemp, tmin, tmax = 40, 0.1, 50.
    nene, emin, emax = 2000, 2., 8.
    metal = 0.3
    spTable = apec_table(nz, zmin, zmax, ntemp, tmin, tmax, nene, emin, emax, metal=metal, tbroad=True, abund='aspl',
                         flag_ene=False, apecroot=(3, 0, 9))
    clear_file(specTableFile)
    write_spectable(spTable, specTableFile, overwrite=True)

    assert_hdu_list_matches_reference(fits.open(specTableFile), fits.open(referenceSpecTableFile), tol=1e-6,
                                      warn_on_keys=True)


@pytest.fixture(scope="module", autouse=True)
def on_end_module():
    yield
    clear_file(specTableFile)
