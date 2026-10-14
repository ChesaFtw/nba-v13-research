"""Check the historical opening odds pilot configuration."""
from scripts.opening_odds_probe import GAME_ODDS_URL, PROP_ODDS_URL

def test_opening_endpoints_are_separate():
    assert GAME_ODDS_URL != PROP_ODDS_URL
    assert GAME_ODDS_URL.endswith('/odds/opening')
    assert PROP_ODDS_URL.endswith('/player_props/opening')
