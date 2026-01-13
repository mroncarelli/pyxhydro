import numpy as np
import pytest

from pyxhydro.observ import ra_corr

from .randomutils import TrueRandomGenerator, globalRandomSeed

TRG = TrueRandomGenerator(globalRandomSeed)
errMsg = "Random seed: " + str(TRG.initialSeed)  # Assertion error message if test fails

rad2deg = 180 / np.pi
rad2arcmin = rad2deg * 60

# Random angle in full circle
ra_full = TRG.uniform(0, 2 * np.pi)  # [rad]

# Random angle in first and second half circles
ra_half1 = TRG.uniform(0, np.pi)  # [deg]
ra_half2 = ra_half1 + np.pi  # [deg]

ra_tuple = (ra_full, ra_half1, ra_half2)
ra_list = list(ra_tuple)
ra_array = np.asarray(ra_tuple)

def test_angle_between_zero_and_twopi_must_remain_the_same():
    """
    An input angle in the interval [0, 2pi[ must remain unchanged
    """
    assert ra_full == pytest.approx(ra_corr(ra_full)), errMsg


def test_result_must_be_the_same_when_changing_units():
    """
    Result must not change when changing units from rad to deg o arcmin
    """
    ref_full = ra_corr(ra_full)
    assert ra_corr(ra_full * rad2deg, units='deg') / rad2deg == pytest.approx(ref_full), errMsg
    assert ra_corr(ra_full * rad2arcmin, units='arcmin') / rad2arcmin == pytest.approx(ref_full), errMsg

    ref_half1 = ra_corr(ra_half1)
    assert ra_corr(ra_half1 * rad2deg, units='deg') / rad2deg == pytest.approx(ref_half1), errMsg
    assert ra_corr(ra_half1 * rad2arcmin, units='arcmin') / rad2arcmin == pytest.approx(ref_half1), errMsg

    ref_half2 = ra_corr(ra_half2)
    assert ra_corr(ra_half2 * rad2deg, units='deg') / rad2deg == pytest.approx(ref_half2), errMsg
    assert ra_corr(ra_half2 * rad2arcmin, units='arcmin') / rad2arcmin == pytest.approx(ref_half2), errMsg


def test_result_must_be_periodic_with_twopi():
    """
    Result must change when adding 2pi
    """
    ref_full = ra_corr(ra_full)
    for index in range(-10, 10):
        assert ra_corr(ra_full + index * 2 * np.pi) == pytest.approx(ref_full), errMsg


def test_ra_in_first_half_circle_must_remain_the_same_with_center_zero():
    """
    With zero = True and angle in the first half circle must remain unchanged
    """
    assert ra_corr(ra_half1, zero=True) == pytest.approx(ra_half1), errMsg


def test_ra_in_second_half_circle_must_change_to_opposite_angle():
    """
    With zero = True and angle in the first half circle must remain unchanged
    """
    assert ra_corr(ra_half2, zero=True) == pytest.approx(-2 * np.pi + ra_half2), errMsg


def test_result_must_match_input_type():
    """
    Result must match the input type both with zero=True and False
    """
    inp_list = [ra_full, ra_tuple, ra_list, ra_array]
    for inp in inp_list:
        assert type(ra_corr(inp, zero=False)) == type(inp), errMsg
        assert type(ra_corr(inp, zero=True)) == type(inp), errMsg


def test_result_in_iterable_must_match_the_one_from_value():
    """
    Results obtained using an iterable as an argument must match the ones obtained with the items of the iterable.
    """
    inp_list = [ra_tuple, ra_list, ra_array]
    for inp in inp_list:
        result = ra_corr(inp, zero=False)
        for item1, item2 in zip(result, inp):
            assert item1 == pytest.approx(ra_corr(item2, zero=False)), errMsg
        result = ra_corr(inp, zero=True)
        for item1, item2 in zip(result, inp):
            assert item1 == pytest.approx(ra_corr(item2, zero=True)), errMsg
