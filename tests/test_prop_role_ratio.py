from research.prop_role_diagnostics import error_ratio


def test_simple_ratio():
    assert error_ratio([2, 4], [1, 2]) == 2


def test_equal_error():
    assert error_ratio([1, 1], [1, 1]) == 1
