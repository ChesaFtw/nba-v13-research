import os
from v13.player_history import fetch_pages


class Response:
    status_code = 200
    headers = {}
    def raise_for_status(self):
        pass
    def __init__(self, data, cursor):
        self.data, self.cursor = data, cursor
    def json(self):
        return {"data": self.data, "meta": {"next_cursor": self.cursor}}


class Session:
    def __init__(self):
        self.calls = []
    def get(self, url, headers, params, timeout):
        self.calls.append(params.copy())
        return Response([{"id": len(self.calls)}], 123 if len(self.calls) == 1 else None)


def test_pagination(monkeypatch):
    monkeypatch.setenv("BALLDONTLIE_API_KEY", "test")
    session = Session()
    pages = list(fetch_pages(2025, max_pages=4, session=session))
    assert len(pages) == 2
    assert session.calls[1]["cursor"] == 123


def test_page_cap(monkeypatch):
    monkeypatch.setenv("BALLDONTLIE_API_KEY", "test")
    assert len(list(fetch_pages(2025, max_pages=1, session=Session()))) == 1
