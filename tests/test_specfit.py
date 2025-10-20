import os
import sys

sys.path.append(os.environ.get("HEADAS") + "/lib/python")
# TODO: the lines above are necessary only to make the code work in IntelliJ (useful for debugging)

import pytest

from .fitstestutils import assert_hdu_list_matches_reference
from .specfittestutils import *
from xraysim.specutils.specfit import *

input_dir = os.environ.get('XRAYSIM') + '/tests/inp/'
reference_dir = os.environ.get('XRAYSIM') + '/tests/reference_files/'
spectrumApec = input_dir + 'apec_fakeit_for_test.pha'
spectrumBapec = input_dir + 'bapec_fakeit_for_test.pha'
spectrumApecNoStat = input_dir + 'apec_fakeit_nostat_for_test.pha'
spectrumBapecNoStat = input_dir + 'bapec_fakeit_nostat_for_test.pha'
bapecSpecFitFile = input_dir +"bapec_specfit_created_for_test.spf"
specFitReferenceFile = reference_dir + "reference_bapec_wrong_pars.spf"

instrumentDir = os.environ.get("SIXTE") + "/share/sixte/instruments/xrism-resolve-test/"
rmf = instrumentDir + "rsl_Hp_5eV.rmf"
arf = instrumentDir + "rsl_sixte_standard_GVclosed.arf"
rightParsApec = (7., 0.2, 0.15, 0.1)
rightParsBapec = (5., 0.3, 0.2, 300., 0.1)
wrongParsApec = (3., 0.4, 0.15, 2.)  # redshift is correct
wrongParsBapec = (1., 0.1, 0.2, 100., 4.)  # redshift is correct
toleranceNoStat = 0.1  # tolerance when starting with correct redshift and no statistical fluctuations
toleranceWithStat = 1.4  # tolerance when starting with correct redshift


def fit_test(spectrum: str, model: str, start: tuple, method: str, reference: tuple, tolerance: float):
    """
    Generic test that checks that a fit is done with no errors and results are within tolerance
    :param spectrum: (str) Spectrum file
    :param model: (str) Model name
    :param start: (tuple of float) Start parameters
    :param method: (str) Fit method
    :param reference: (tuple of float) Reference values to check
    :param tolerance: (float) Relative tolerance in the results
    :return:
    """
    specfit = SpecFit(spectrum, model, respFile=rmf, arfFile=arf)
    specfit.run(start=start, method=method)
    assert_specfit_has_no_error_flags(specfit)
    assert_fit_results_within_tolerance(specfit, reference, tol=tolerance)


def test_apec_no_stat_fit_start_with_right_parameters():
    # Fitting the apec spectrum produced with fakeit and no statistical fluctuations starting with the right parameters
    # should lead to the correct result, within tolerance
    fit_test(spectrumApecNoStat, 'apec', rightParsApec, 'cstat', rightParsApec, toleranceNoStat)


def test_bapec_no_stat_fit_start_with_right_parameters():
    # Fitting the bapec spectrum produced with fakeit and no statistical fluctuations starting with the right
    # parameters should lead to the correct result, within tolerance
    fit_test(spectrumBapecNoStat, 'bapec', rightParsBapec, 'cstat', rightParsBapec, toleranceNoStat)


def test_apec_fit_start_with_right_parameters():
    # Fitting the apec spectrum produced with fakeit starting with the right parameters should
    # lead to the correct result, within tolerance
    fit_test(spectrumApec, 'apec', rightParsApec, 'cstat', rightParsApec, toleranceWithStat)


def test_bapec_fit_start_with_right_parameters():
    # Fitting the bapec spectrum produced with fakeit starting with the right parameters should
    # lead to the correct result, within tolerance
    fit_test(spectrumBapec, 'bapec', rightParsBapec, 'cstat', rightParsBapec, toleranceWithStat)


def test_fit_two_spectra_start_with_right_parameters():
    # Fitting the apec and bapec spectrum produced with fakeit, one after the other, starting with the right
    # parameters should lead to the correct result, within tolerance. Spectra are fitted in inverse order with respect
    # to their loading to ensure that the fitting procedure works even in this case. It also checks that the xspec
    # state variables are restored correctly after the fit.
    specfit_apec = SpecFit(spectrumApec, "apec", respFile=rmf, arfFile=arf)
    assert not specfit_apec.fitDone
    assert_specfit_has_coherent_properties(specfit_apec)
    specfit_bapec = SpecFit(spectrumBapec, "bapec", respFile=rmf, arfFile=arf)
    assert not specfit_bapec.fitDone
    assert_specfit_has_coherent_properties(specfit_bapec)

    n_spectra = xsp.AllData.nSpectra
    noticed1 = xsp.AllData(1).noticed
    noticed2 = xsp.AllData(2).noticed
    active_model = xsp.AllModels.sources[1]

    specfit_bapec.run(start=rightParsBapec, method="cstat")
    assert specfit_bapec.fitDone
    assert_specfit_has_coherent_properties(specfit_bapec)

    assert xsp.AllData.nSpectra == n_spectra
    assert xsp.AllData(1).noticed == noticed1
    assert xsp.AllData(2).noticed == noticed2
    assert xsp.AllModels.sources[1] == active_model
    assert_fit_results_within_tolerance(specfit_bapec, rightParsBapec, tol=toleranceWithStat)

    specfit_apec.run(start=rightParsApec, method="cstat")
    assert specfit_apec.fitDone
    assert_specfit_has_coherent_properties(specfit_apec)

    assert xsp.AllData.nSpectra == n_spectra
    assert xsp.AllData(1).noticed == noticed1
    assert xsp.AllData(2).noticed == noticed2
    assert xsp.AllModels.sources[1] == active_model
    assert_fit_results_within_tolerance(specfit_apec, rightParsApec, tol=toleranceWithStat)


def test_apec_no_stat_fit_start_with_only_redshift_right():
    # Fitting the apec spectrum produced with fakeit and no statistical fluctuations, starting with all wrong
    # parameters except for redshift should lead to the correct result, within tolerance
    fit_test(spectrumApecNoStat, 'apec', wrongParsApec, 'cstat', rightParsApec, toleranceNoStat)


def test_bapec_no_stat_fit_start_with_only_redshift_right():
    # Fitting the bapec spectrum produced with fakeit and no statistical fluctuations, starting with all wrong
    # parameters except for redshift should lead to the correct result, within tolerance
    fit_test(spectrumBapecNoStat, 'bapec', wrongParsBapec, 'cstat', rightParsBapec, toleranceNoStat)


def test_apec_fit_start_with_only_redshift_right():
    # Fitting the apec spectrum produced with fakeit, starting with all wrong parameters except for redshift should
    # lead to the correct result, within tolerance
    fit_test(spectrumApec, 'apec', wrongParsApec, 'cstat', rightParsApec, toleranceWithStat)


def test_covariance_and_correlation_matrices_are_none_at_initialization():
    """
    Tests that the covariance and the correlation matrices are None before running the fit
    """
    specfit = SpecFit(spectrumBapec, "bapec", respFile=rmf, arfFile=arf)
    assert specfit.covariance_matrix() is None
    assert specfit.correlation_matrix() is None


specFitBapec = SpecFit(spectrumBapec, "bapec", respFile=rmf, arfFile=arf)
specFitBapec.run(start=wrongParsBapec, method="cstat")


def test_bapec_fit_start_with_only_redshift_right():
    # Fitting the bapec spectrum produced with fakeit, starting with all wrong parameters except for redshift should
    # lead to the correct result, within tolerance
    assert_fit_results_within_tolerance(specFitBapec, rightParsBapec, tol=toleranceWithStat)


def test_covariance_matrix_has_correct_shape_and_diagonal_elements():
    """
    After the fit has been done the covariance matrix should have a shape equal to the number of free parameters per
    side and the diagonal elements should correspond to the product of the fit errors for the corresponding parameters.
    """
    covariance_matrix = specFitBapec.covariance_matrix()
    assert covariance_matrix.shape == (specFitBapec.nFree, specFitBapec.nFree)
    for i in range(specFitBapec.nFree):
        assert covariance_matrix[i, i] == pytest.approx(specFitBapec.fitResult["sigma"][i] ** 2)

# Calculating correlation matrix
correlationMatrix = specFitBapec.correlation_matrix()


def test_correlation_matrix_has_correct_shape_and_diagonal_elements():
    """
    After the fit has been done the correlation matrix should have a shape equal to the number of free parameters per
    side and the diagonal elements should be equal to 1.
    """
    assert correlationMatrix.shape == (specFitBapec.nFree, specFitBapec.nFree)
    for i in range(specFitBapec.nFree):
        assert correlationMatrix[i, i] == pytest.approx(1)


def test_correlation_matrix_has_all_values_between_minus_one_and_one():
    """
    After the fit has been done the correlation matrix should contain only values between -1 and 1.
    """
    check = np.ndarray(correlationMatrix.shape, dtype=bool)
    for i, j in np.ndindex(check.shape):
        check[i, j] = correlationMatrix[i, j] == pytest.approx(-1) or correlationMatrix[i, j] > -1
    assert check.all()

    for i, j in np.ndindex(check.shape):
        check[i, j] = correlationMatrix[i, j] == pytest.approx(1) or correlationMatrix[i, j] < 1
    assert check.all()

if os.path.isfile(bapecSpecFitFile):
    os.remove(bapecSpecFitFile)
specFitBapec.save(bapecSpecFitFile, overwrite=True)


def test_specfit_file_has_been_created_and_matches_reference():
    assert os.path.isfile(bapecSpecFitFile)
    assert_hdu_list_matches_reference(fits.open(bapecSpecFitFile), fits.open(specFitReferenceFile), tol=1e-4)
    os.remove(bapecSpecFitFile)

