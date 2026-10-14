import pytest
from v13.core import american_to_decimal, expected_return, implied_probability

def test_american_odds():
    assert american_to_decimal(-110) == pytest.approx(1.9090909)
    assert american_to_decimal(150) == 2.5

def test_break_even():
    d = american_to_decimal(-110)
    assert expected_return(implied_probability(d), d) == pytest.approx(0)

def test_bad_inputs():
    with pytest.raises(ValueError):
        implied_probability(1)
    with pytest.raises(ValueError):
        expected_return(1.2, 2)
