"""
Set of methods useful to compare SpecFit instances
"""
import pytest
import warnings

from xraysim.specutils.specfit import SpecFit


def assert_specfit_has_coherent_properties(specfit: SpecFit, msg='') -> None:
    """
    Checks that a SpecFit object has coherent properties
    :param specfit: (SpecFit)
    :param msg: (str) If provided it is written in the assertion error message
    :return: None
    """
    assert specfit.nParameters > 0, msg
    assert len(specfit.parNames) == specfit.nParameters, msg
    assert type(specfit.fitDone) == bool, msg
    if specfit.fitDone:
        assert specfit.nFixed + specfit.nFree == specfit.nParameters, msg
        for ipar in range(specfit.nParameters):
            assert type(specfit.parFixed[ipar]) == bool, msg
            assert type(specfit.parFree[ipar]) == bool, msg
            assert specfit.parFixed[ipar] != specfit.parFree[ipar], msg

        assert len(specfit.fixedParNames) == specfit.nFixed, msg
        assert len(specfit.freeParNames) == specfit.nFree, msg
        assert set(specfit.fixedParNames).union(set(specfit.freeParNames)) == set(specfit.parNames), msg
        assert hasattr(specfit, 'fitResult'), msg
        assert hasattr(specfit, 'fitPoints'), msg
    else:
        assert specfit.nFixed is None, msg
        assert specfit.nFree is None, msg
        assert specfit.parFixed is None, msg
        assert specfit.parFree is None, msg
        assert specfit.fixedParNames is None, msg
        assert specfit.freeParNames is None, msg
        assert not hasattr(specfit, 'fitResult'), msg
        assert not hasattr(specfit, 'fitPoints'), msg

    return


def assert_fit_results_within_error(specfit: SpecFit, reference, sigma_tol=1, rel=0, msg='') -> None:
    """
    Checks that a SpecFit result matches the reference values within tolerance based on the error of the fit. An
    additional relative tolerance based on the absolute value may be added.
    :param specfit: (SpecFit) SpeFit containing the fit results
    :param reference: (float, tuple) Reference values
    :param sigma_tol: Tolerance in units of statistical error
    :param rel: Tolerance in units of the reference values
    :param msg: (str) If provided it is written in the assertion error message
    :return: None
    """

    assert specfit.model.nParameters == len(reference), msg

    msg_ = ', ' + msg if type(msg) == str and len(msg) > 0 else ''
    for index, val in enumerate(reference):
        ind_fit = specfit.model.startParIndex + index
        if not specfit.model(ind_fit).frozen:
            total_tolerance = sigma_tol * specfit.model(ind_fit) + rel * val
            assert abs(specfit.model(ind_fit).values[0] - val) < total_tolerance, \
                ("Fit result: " + str(specfit.model(ind_fit).values[0]) + ", Reference: " + str(val) +
                 ", Tolerance: " + str(total_tolerance) + msg_)


def assert_fit_results_nominal_within_tolerance(specfit: SpecFit, reference, tol=1., msg='') -> None:
    """
    Checks that a SpecFit result nominal values match the reference values within tolerance
    :param specfit: (SpecFit) SpeFit containing the fit results
    :param reference: (float, tuple) Reference values
    :param tol: Relative tolerance
    :param msg: (str) If provided it is written in the assertion error message
    :return: None
    """


    assert specfit.model.nParameters == len(reference), msg

    msg_ = ', ' + msg if type(msg) == str and len(msg) > 0 else ''
    for index, val in enumerate(reference):
        ind_fit = specfit.model.startParIndex + index
        if not specfit.model(ind_fit).frozen:
            assert specfit.model(ind_fit).values[0] == pytest.approx(val, rel=tol), \
                ("Fit result: " + str(specfit.model(ind_fit).values[0]) + ", Reference: " + str(val))


def assert_specfit_has_no_error_flags(specfit: SpecFit, msg='') -> None:
    """
    Checks that a SpecFit instance has no error flags in the fit results
    :param specfit: (SpecFit) SpeFit containing the fit results
    :param msg: (str) If provided it is written in the assertion error message
    :return: None
    """

    msg_ = ', ' + msg if type(msg) == str and len(msg) > 0 else ''
    if specfit.fitDone or specfit.restored:
        for index, par in enumerate(specfit.freeParNames):
            if specfit.parFree[index]:
                assert 'T' not in specfit.fitResult["error_flags"][index], "Parameter " + par + " has errors" + msg_
            else:
                assert specfit.fitResult["error_flags"][index] == '', ("Parameter " + par + " (fixed) has error flags"
                                                                       + msg_)
    else:
        warnings.warn("Can not check error flags because the fit has not been run.")

    return None
