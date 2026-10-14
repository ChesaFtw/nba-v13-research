import os
from unittest.mock import patch
from v13.history import fetch_season

class FakeResponse:
    status_code = 200
    def __init__(self, data, cursor):
        self.data, self.cursor = data, cursor
    def raise_for_status(self):
        pass
    def json(self):
        return {"data": self.data, "meta": {"next_cursor": self.cursor}}

class FakeSession:
    def __init__(self):
        self.calls = []
    def get(self, url, headers, params, timeout):
        self.calls.append(dict(params))
        if len(self.calls) == 1:
            return FakeResponse([{"id": 2, "date": "2024-10-24"}], 10)
        return FakeResponse([{"id": 1, "date": "2024-10-22"}], None)

def test_historical_pagination():
    client = FakeSession()
    with patch.dict(os.environ, {"BALLDONTLIE_API_KEY": "dummy"}):
        games = fetch_season(2024, session=client)
    assert [g["id"] for g in games] == [1, 2]
    assert client.calls[0]["season_type"] == "regular"
    assert client.calls[1]["cursor"] == 10
