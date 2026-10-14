"""Audit four completed regular-season model reports before promotion."""
import argparse
import json
from pathlib import Path

SEASONS = (2022, 2023, 2024, 2025)
MARKETS = ("points", "rebounds", "assists", "threes", "pra")

def summarize(reports):
    if set(reports) != set(SEASONS):
        raise ValueError("Require all four seasons")
    result = {}
    for season, report in reports.items():
        if report["season"] != season:
            raise ValueError("Season mismatch")
        if report["regular_games"] != 1230:
            raise ValueError("Incomplete regular-season coverage")
        if report["holdout_rows"] < 100:
            raise ValueError("Insufficient holdout")
        if set(report["markets"]) != set(MARKETS):
            raise ValueError("Missing markets")
    for market in MARKETS:
        total = sum(r["holdout_rows"] for r in reports.values())
        baseline = sum(r["holdout_rows"] * r["markets"][market]["rolling_holdout_mae"]
                       for r in reports.values()) / total
        ridge = sum(r["holdout_rows"] * r["markets"][market]["ridge_holdout_mae"]
                    for r in reports.values()) / total
        wins = sum(r["markets"][market]["ridge_beats_selected_rolling"]
                   for r in reports.values())
        result[market] = {"holdout_rows": total,
                          "weighted_rolling_mae": round(baseline, 5),
                          "weighted_ridge_mae": round(ridge, 5),
                          "ridge_season_wins": wins,
                          "promote_ridge": ridge < baseline and wins >= 3}
    return {"markets": result,
            "scope": "Regular-season within-season chronological holdouts",
            "limitations": "MAE is not betting win rate; no actual betting lines"}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    reports = {season: json.loads((Path(a.input) /
               f"regular_model_{season}.json").read_text()) for season in SEASONS}
    result = summarize(reports)
    Path(a.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
