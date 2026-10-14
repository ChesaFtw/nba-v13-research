"""Backfill historical games as JSONL files for reproducible research."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from v13.history import fetch_season

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", type=int, nargs="+", default=[2022, 2023, 2024, 2025])
    parser.add_argument("--out", default="research_data")
    args = parser.parse_args()
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    manifest = {"retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                "source": "BALLDONTLIE /v1/games", "season_type": "regular", "seasons": {}}
    for season in args.seasons:
        games = fetch_season(season)
        if len(games) < 100:
            raise RuntimeError(f"Season {season} returned only {len(games)} games; refusing incomplete dataset")
        file = root / f"nba_games_regular_{season}.jsonl"
        with file.open("w", encoding="utf-8") as output:
            for game in games:
                output.write(json.dumps(game, separators=(",", ":")) + "\n")
        manifest["seasons"][str(season)] = {"count": len(games), "file": file.name,
            "earliest_date": games[0].get("date"), "latest_date": games[-1].get("date")}
        print(f"{season}: {len(games)} games -> {file}", flush=True)
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))

if __name__ == "__main__":
    main()
