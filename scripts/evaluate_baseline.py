"""Evaluate the chronological Elo baseline from archived historical game files."""
import argparse
import json
from pathlib import Path
from v13.baseline import evaluate_seasons, summarize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="research_data")
    parser.add_argument("--out", default="research_results")
    args = parser.parse_args()
    data = Path(args.data)
    seasons = []
    for path in sorted(data.glob("nba_games_regular_*.jsonl")):
        season = int(path.stem.rsplit("_", 1)[-1])
        games = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if len(games) < 1000:
            raise RuntimeError(f"Incomplete historical season {season}: {len(games)} games")
        seasons.append((season, games))
    if len(seasons) < 4:
        raise RuntimeError(f"Expected at least 4 seasons; found {len(seasons)}")
    records = evaluate_seasons(seasons)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "elo_predictions.jsonl").open("w") as f:
        for row in records:
            f.write(json.dumps(row) + "\n")
    report = {
        "model": "chronological_elo_v1",
        "description": "Pregame winner baseline; no market odds or player data; not confidence calibrated",
        "seasons": {str(s): summarize([r for r in records if r["season"] == s]) for s, _ in seasons},
        "overall": summarize(records),
        "validation_warning": "In-sample sequential updates, NOT independent holdout validation; do not claim predictive edge."
    }
    (output / "baseline_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
