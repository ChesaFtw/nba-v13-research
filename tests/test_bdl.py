from unittest.mock import patch
import pytest
from v13.bdl import get_games

def test_preseason_filter_sent():
    class Response:
        def raise_for_status(self):
            pass
        def json(self):
            return {"data": [{"id": 1}], "meta": {"next_cursor": None}}
    with patch.dict("os.environ", {"BALLDONTLIE_API_KEY": "test-key"}):
        with patch("v13.bdl.requests.get", return_value=Response()) as get:
            assert get_games(["2026-10-14"], season_type="preseason") == [{"id": 1}]
            assert ("season_type", "preseason") in get.call_args.kwargs["params"]
            assert ("dates[]", "2026-10-14") in get.call_args.kwargs["params"]

def test_invalid_season_type():
    with patch.dict("os.environ", {"BALLDONTLIE_API_KEY": "test-key"}):
        with pytest.raises(ValueError):
            get_games(["2026-10-14"], season_type="unknown")
