import unittest
from scripts.backfill_games import fetch_games

class Response:
    status_code = 200
    def __init__(self, payload):
        self.payload = payload
    def json(self):
        return self.payload
    def raise_for_status(self):
        return None

class Session:
    def __init__(self, pages):
        self.pages = iter(pages)
    def get(self, *args, **kwargs):
        return Response(next(self.pages))

class GameIndexTests(unittest.TestCase):
    def test_pagination(self):
        import os
        previous = os.environ.get("BALLDONTLIE_API_KEY")
        os.environ["BALLDONTLIE_API_KEY"] = "test-only"
        try:
            session = Session([
                {"data": [{"id": 1}], "meta": {"next_cursor": 9}},
                {"data": [{"id": 2}], "meta": {"next_cursor": None}},
            ])
            self.assertEqual(len(fetch_games(2024, session=session, sleep=lambda _: None)), 2)
        finally:
            if previous is None:
                os.environ.pop("BALLDONTLIE_API_KEY", None)
            else:
                os.environ["BALLDONTLIE_API_KEY"] = previous

if __name__ == "__main__":
    unittest.main()
