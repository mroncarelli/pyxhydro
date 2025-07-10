"""
Set of methods useful to compare SpecFit instances
"""
import warnings

from xraysim.specutils.specfit import SpecFit


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
    Checks that a Xspec model containing a fit result matches the reference values within tolerance
    :param specfit: (SpecFit) SpeFit containing the fit results
    :param reference: (float, tuple) Reference values
    :param tol: Tolerance in units of statistical error
    :return: None
    """

    assert specfit.model.nParameters == len(reference)
    for index, val in enumerate(reference):
        ind_fit = specfit.model.startParIndex + index
        if not specfit.model(ind_fit).frozen:
            assert abs(specfit.model(ind_fit).values[0] - val) < tol * specfit.model(ind_fit).sigma


def assert_specfit_has_no_error_flags(specfit: SpecFit) -> None:
    """
    Checks that a SpecFit instance has no error flags in the fit results
    :param specfit: (SpecFit) SpeFit containing the fit results
    :return: None
    """
    if specfit.fitDone or specfit.restored:
        for index, par in enumerate(specfit.freeParNames):
            if specfit.parFree[index]:
                assert specfit.fitResult["error_flags"][index] == 'FFFFFFFFF', "Parameter " + par + " has errors"
            else:
                assert specfit.fitResult["error_flags"][index] == '', "Parameter " + par + " (fixed) has errors flags"
    else:
        warnings.warn("Can not check error flags because the fit has not been run.")

    return None
