import unittest
from scripts.walk_forward_prop_mae import evaluate
from v13.prop_features import MARKETS

class WalkForwardTests(unittest.TestCase):
    def rows(self):
        result = []
        for day in range(90):
            for player in range(6):
                row = {"player_id": player, "game_date": f"2025-{1+day//28:02d}-{1+day%28:02d}"}
                for market in MARKETS:
                    for window in (5, 10, 20):
                        row[f"{market}_avg_{window}"] = float(player + day % 7)
                    row[f"{market}_actual"] = float(player + day % 7)
                result.append(row)
        return result

    def test_identical_holdouts_and_no_future_training(self):
        report = evaluate(self.rows(), min_train_dates=40, test_dates=10)
        counts = set()
        for market in MARKETS:
            m = report["markets"][market]
            counts.add(m["n"])
            self.assertAlmostEqual(m["last10_mae"], 0)
            self.assertTrue(all(f["train_before"] <= f["test_through"] for f in m["folds"]))
        self.assertEqual(len(counts), 1)

    def test_duplicate_player_date_rejected(self):
        rows = self.rows()
        with self.assertRaises(ValueError):
            evaluate(rows + [rows[0]], min_train_dates=40)

if __name__ == "__main__":
    unittest.main()
