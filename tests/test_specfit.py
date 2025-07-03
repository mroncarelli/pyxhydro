import os
import sys
sys.path.append(os.environ.get("HEADAS") + "/lib/python")
# TODO: the lines above are necessary only to make the code work in IntelliJ (useful for debugging)

import pytest

from xraysim.specutils.specfit import *

input_dir = os.environ.get('XRAYSIM') + '/tests/inp/'
spectrumApec = input_dir + 'apec_fakeit_for_test.pha'
spectrumBapec = input_dir + 'bapec_fakeit_for_test.pha'
spectrumApecNoStat = input_dir + 'apec_fakeit_nostat_for_test.pha'
spectrumBapecNoStat = input_dir + 'bapec_fakeit_nostat_for_test.pha'
rmf = input_dir + 'resolve_h5ev_2019a.rmf'
arf = input_dir + 'resolve_pnt_heasim_noGV_20190701.arf'
right_pars_apec = [7., 0.2, 0.15, 0.1]
right_pars_bapec = [5., 0.3, 0.2, 300., 0.1]
wrong_pars_apec = [3., 0.4, 0.05, 2.]
wrong_pars_bapec = [1., 0.1, 0.6, 100., 4.]
toleranceNoStat = 0.1  # tolerance when starting with correct redshift and no statistical fluctuations
tolerance = 1.4  # tolerance when starting with correct redshift


def assert_specfit_has_coherent_properties(specfit: SpecFit) -> None:
    """
    Checks that a SpecFit object has coherent properties
    :param specfit: (SpecFit)
    :return: None
    """
    assert specfit.nParameters > 0
    assert len(specfit.parNames) == specfit.nParameters
    assert type(specfit.fitDone) == bool, "fitDone property must be True or False"
    if specfit.fitDone:
        assert specfit.nFixed + specfit.nFree == specfit.nParameters
        for ipar in range(specfit.nParameters):
            assert type(specfit.parFixed[ipar]) == bool
            assert type(specfit.parFree[ipar]) == bool
            assert specfit.parFixed[ipar] != specfit.parFree[ipar]

        assert len(specfit.fixedParNames) == specfit.nFixed
        assert len(specfit.freeParNames) == specfit.nFree
        assert set(specfit.fixedParNames).union(set(specfit.freeParNames)) == set(specfit.parNames)
        assert hasattr(specfit, 'fitResult')
        assert hasattr(specfit, 'fitPoints')
    else:
        assert specfit.nFixed is None
        assert specfit.nFree is None
        assert specfit.parFixed is None
        assert specfit.parFree is None
        assert specfit.fixedParNames is None
        assert specfit.freeParNames is None
        assert not hasattr(specfit, 'fitResult')
        assert not hasattr(specfit, 'fitPoints')

    return


def assert_fit_results_within_tolerance(specfit: SpecFit, reference, tol=1.) -> None:
    """
    Checks that a Xspec model containing a fit result matches the reference values within
    tolerance
    :param specfit: (SpecFit) Xspec model containing the fit results
    :param reference: (float, tuple) Reference values
    :param tol: Tolerance in units of statistical error
    :return: None
    """

    assert specfit.model.nParameters == len(reference)
    for ind, val in enumerate(reference):
        ind_fit = specfit.model.startParIndex + ind
        if not specfit.model(ind_fit).frozen:
            assert abs(specfit.model(ind_fit).values[0] - val) < tol * specfit.model(ind_fit).sigma


def test_apec_no_stat_fit_start_with_right_parameters():
    # Fitting the apec spectrum produced with fakeit and no statistical fluctuations starting with the right parameters
    # should lead to the correct result, within tolerance
    specfit = SpecFit(spectrumApecNoStat, "apec", rmf=rmf, arf=arf)
    specfit.run(start=right_pars_apec, method="cstat")
    assert_fit_results_within_tolerance(specfit, right_pars_apec, tol=toleranceNoStat)


def test_bapec_no_stat_fit_start_with_right_parameters():
    # Fitting the bapec spectrum produced with fakeit and no statistical fluctuations starting with the right
    # parameters should lead to the correct result, within tolerance
    specfit = SpecFit(spectrumBapecNoStat, "bapec", rmf=rmf, arf=arf)
    specfit.run(start=right_pars_bapec, method="cstat")
    assert_fit_results_within_tolerance(specfit, right_pars_bapec, tol=toleranceNoStat)


def test_apec_fit_start_with_right_parameters():
    # Fitting the apec spectrum produced with fakeit starting with the right parameters should
    # lead to the correct result, within tolerance
    specfit = SpecFit(spectrumApec, "apec", rmf=rmf, arf=arf)
    specfit.run(start=right_pars_apec, method="cstat")
    assert_fit_results_within_tolerance(specfit, right_pars_apec, tol=tolerance)


def test_bapec_fit_start_with_right_parameters():
    # Fitting the bapec spectrum produced with fakeit starting with the right parameters should
    # lead to the correct result, within tolerance
    specfit = SpecFit(spectrumBapec, "bapec", rmf=rmf, arf=arf)
    specfit.run(start=right_pars_bapec, method="cstat")
    assert_fit_results_within_tolerance(specfit, right_pars_bapec, tol=tolerance)


def test_fit_two_spectra_start_with_right_parameters():
    # Fitting the apec and bapec spectrum produced with fakeit, one after the other, starting with the right
    # parameters should lead to the correct result, within tolerance. Spectra are fitted in inverse order with respect
    # to their loading to ensure that the fitting procedure works even in this case. It also checks that the xspec
    # state variables are restored correctly after the fit.
    specfit_apec = SpecFit(spectrumApec, "apec", rmf=rmf, arf=arf)
    assert not specfit_apec.fitDone
    assert_specfit_has_coherent_properties(specfit_apec)
    specfit_bapec = SpecFit(spectrumBapec, "bapec", rmf=rmf, arf=arf)
    assert not specfit_bapec.fitDone
    assert_specfit_has_coherent_properties(specfit_bapec)

    n_spectra = xsp.AllData.nSpectra
    noticed1 = xsp.AllData(1).noticed
    noticed2 = xsp.AllData(2).noticed
    active_model = xsp.AllModels.sources[1]

    specfit_bapec.run(start=right_pars_bapec, method="cstat")
    assert specfit_bapec.fitDone
    assert_specfit_has_coherent_properties(specfit_bapec)

    assert xsp.AllData.nSpectra == n_spectra
    assert xsp.AllData(1).noticed == noticed1
    assert xsp.AllData(2).noticed == noticed2
    assert xsp.AllModels.sources[1] == active_model
    assert_fit_results_within_tolerance(specfit_bapec, right_pars_bapec, tol=tolerance)

    specfit_apec.run(start=right_pars_apec, method="cstat")
    assert specfit_apec.fitDone
    assert_specfit_has_coherent_properties(specfit_apec)

    assert xsp.AllData.nSpectra == n_spectra
    assert xsp.AllData(1).noticed == noticed1
    assert xsp.AllData(2).noticed == noticed2
    assert xsp.AllModels.sources[1] == active_model
    assert_fit_results_within_tolerance(specfit_apec, right_pars_apec, tol=tolerance)


def test_apec_no_stat_fit_start_with_only_redshift_right():
    # Fitting the apec spectrum produced with fakeit and no statistical fluctuations, starting with all wrong
    # parameters except for redshift should lead to the correct result, within tolerance
    specfit = SpecFit(spectrumApecNoStat, "apec", rmf=rmf, arf=arf)
    start_pars = wrong_pars_apec
    start_pars[2] = right_pars_apec[2]
    specfit.run(start=start_pars, method="cstat")
    assert_fit_results_within_tolerance(specfit, right_pars_apec, tol=toleranceNoStat)


def test_bapec_no_stat_fit_start_with_only_redshift_right():
    # Fitting the bapec spectrum produced with fakeit and no statistical fluctuations, starting with all wrong
    # parameters except for redshift should lead to the correct result, within tolerance
    specfit = SpecFit(spectrumBapecNoStat, "bapec", rmf=rmf, arf=arf)
    start_pars = wrong_pars_bapec
    start_pars[2] = right_pars_bapec[2]
    specfit.run(start=start_pars, method="cstat")
    assert_fit_results_within_tolerance(specfit, right_pars_bapec, tol=toleranceNoStat)


def test_apec_fit_start_with_only_redshift_right():
    # Fitting the apec spectrum produced with fakeit, starting with all wrong parameters except for redshift should
    # lead to the correct result, within tolerance
    specfit = SpecFit(spectrumApec, "apec", rmf=rmf, arf=arf)
    start_pars = wrong_pars_apec
    start_pars[2] = right_pars_apec[2] 
    specfit.run(start=start_pars, method="cstat")
    assert_fit_results_within_tolerance(specfit, right_pars_apec, tol=tolerance)


def test_covariance_and_correlation_matrices_are_none_at_initialization():
    """
    Tests that the covariance and the correlation matrices are None before running the fit
    """
    specfit = SpecFit(spectrumBapec, "bapec", rmf=rmf, arf=arf)
    assert specfit.covariance_matrix() is None
    assert specfit.correlation_matrix() is None


specFitBapec = SpecFit(spectrumBapec, "bapec", rmf=rmf, arf=arf)
start_pars = wrong_pars_bapec
start_pars[2] = right_pars_bapec[2]
specFitBapec.run(start=start_pars, method="cstat")


def test_bapec_fit_start_with_only_redshift_right():
    # Fitting the bapec spectrum produced with fakeit, starting with all wrong parameters except for redshift should
    # lead to the correct result, within tolerance
    assert_fit_results_within_tolerance(specFitBapec, right_pars_bapec, tol=tolerance)


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


def test_fitpoints_match_with_xspec_plot_values():
    """
    The values saved in the fitPoints attribute must match the ones in the xspec.Plot object. WARNINGS 1) Works only
    if the fit was the latest run. 2) This test has to run last of the set, or it may break the others since it has to
    delete all spectra and models before running.
    """
    xsp.AllData.clear()
    xsp.AllModels.clear()
    specfit = SpecFit(spectrumApec, "apec", rmf=rmf, arf=arf)
    specfit.run(start=right_pars_apec, method="cstat")
    xsp.Plot.xAxis = "keV"
    xsp.Plot("data")
    assert specfit.fitPoints["energy"] == pytest.approx(xsp.Plot.x())
    assert specfit.fitPoints["spectrum"] == pytest.approx(xsp.Plot.y())
    assert specfit.fitPoints["error"] == pytest.approx(xsp.Plot.yErr())
    assert specfit.fitPoints["model"] == pytest.approx(xsp.Plot.model())
