import os
from astropy.io import fits
from xraysim.specutils.tables import apec_table, write_spectable

from .fitstestutils import assert_hdu_list_matches_reference

referenceSpecTableFile = os.environ.get('XRAYSIM') + '/tests/reference_files/reference_emission_table.fits'
specTableFile = "emission_table_created_for_test.fits"

def test_create_apec_table_matches_reference():
    """
    The table created with the apec_table procedure should match the reference one.
    """
    nz, zmin, zmax = 6, 0.07, 0.13
    ntemp, tmin, tmax = 40, 0.1, 50.
    nene, emin, emax = 2000, 2., 8.
    metal = 0.3
    spTable = apec_table(nz, zmin, zmax, ntemp, tmin, tmax, nene, emin, emax, metal=metal, tbroad=True, abund='aspl',
                         flag_ene=False)
    if os.path.isfile(specTableFile):
        os.remove(specTableFile)
    write_spectable(spTable, specTableFile, overwrite=True)

    assert_hdu_list_matches_reference(fits.open(specTableFile), fits.open(referenceSpecTableFile), tol=1e-6)
    os.remove(specTableFile)