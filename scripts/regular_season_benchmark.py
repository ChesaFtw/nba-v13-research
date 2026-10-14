"""Reproducible regular-season-only prop projection comparison.

Reuse downloaded player-stat artifacts, download only the small game-ID index.
This is a date-held-out MAE benchmark, not a historical betting backtest.
"""
import argparse
import json
from pathlib import Path

from scripts.regular_season_filter import filter_regular, load
from scripts.benchmark_prop_models import evaluate as evaluate_baselines
from scripts.train_prop_ridge import evaluate as evaluate_ridge
from v13.prop_features import build_prop_features, MARKETS


def run(season, games, player_stats, source_manifest):
    if not source_manifest.get("complete") or source_manifest.get("truncated"):
        raise ValueError("Player-stat artifact is incomplete")
    if source_manifest.get("season") != season:
        raise ValueError("Player-stat artifact season mismatch")
    clean = filter_regular(games, player_stats, season)
    features = list(build_prop_features(clean))
    if len(features) < 10000:
        raise ValueError("Insufficient regular-season feature rows")
    baseline = evaluate_baselines(features, season)
    ridge = evaluate_ridge(features, season)
    if baseline["holdout_rows"] != ridge["holdout_n"]:
        raise AssertionError("Different holdout row counts")
    if baseline["split_date"] != ridge["cutoff_date"]:
        raise AssertionError("Different chronological holdout dates")
    markets = {}
    for market in MARKETS:
        b = baseline["models"][market]
        r = ridge["markets"][market]
        markets[market] = {
            "selected_rolling_model": b["selected_on_train"],
            "rolling_holdout_mae": b["holdout_selected_mae"],
            "ridge_holdout_mae": r["ridge_mae"],
            "ridge_minus_rolling_mae": round(r["ridge_mae"] - b["holdout_selected_mae"], 4),
            "ridge_beats_selected_rolling": r["ridge_mae"] < b["holdout_selected_mae"],
        }
    report = {
        "season": season,
        "source_player_rows": len(player_stats),
        "regular_played_rows": len(clean),
        "regular_games": len({r["game"]["id"] for r in clean}),
        "excluded_rows": len(player_stats) - len(clean),
        "feature_rows": len(features),
        "train_rows": baseline["training_rows"],
        "holdout_rows": baseline["holdout_rows"],
        "holdout_start": baseline["split_date"],
        "markets": markets,
        "validation": "Regular season IDs and dates verified, zero-minute entries excluded, chronological within-season holdout",
        "limitations": "Projection MAE only. No real odds/lines, no betting hit rate, no multi-season walk-forward. Rolling model selected using train only; Ridge fixed alpha=25.",
    }
    return report, clean


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--season", required=True, type=int)
    p.add_argument("--games", required=True)
    p.add_argument("--stats", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", default="research_results/regular_season")
    args = p.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    report, clean = run(args.season, load(args.games), load(args.stats), manifest)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"regular_model_{args.season}.json").write_text(json.dumps(report, indent=2) + "\n")
    with (out / f"regular_players_{args.season}.jsonl").open("w") as f:
        for row in clean:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
