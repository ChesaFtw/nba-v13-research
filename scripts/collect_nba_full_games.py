"""Persist final NBA games and raw PBP; daily GitHub Actions job calls this module.

Usage: python -m scripts.collect_nba_full_games --days-back 3
No paid API key. Final-only collection; missing/unavailable data fails visibly.
"""
import argparse
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from pathlib import Path
import gzip
import hashlib
import json
import sys

from v13.nba_live_ingest import (
    SCHEDULE_URL, GameDataError, fetch_json, schedule_games, endpoints, normalize_game,
)

ROOT = Path("research_data/live_nba")


def _bytes(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def store_final(root: Path, game_id: str, date: str,
                box: dict, play: dict, *, refresh: bool = False):
    normalized = normalize_game(box, play, game_id, date)
    raw = {"boxscore": box, "playbyplay": play}
    payload = _bytes(raw)
    digest = hashlib.sha256(payload).hexdigest()
    day = root / date
    day.mkdir(parents=True, exist_ok=True)
    path = day / f"{game_id}.json"
    raw_path = day / f"{game_id}.raw.json.gz"
    if path.exists() and raw_path.exists():
        prev = json.loads(path.read_text(encoding="utf-8"))
        if prev.get("raw_sha256") == digest and not refresh:
            return "unchanged"
    normalized["raw_sha256"] = digest
    # gzip timestamp fixed for reproducibility
    path.write_bytes(_bytes(normalized) + b"\n")
    with raw_path.open("wb") as out:
        with gzip.GzipFile(fileobj=out, mode="wb", mtime=0) as compressed:
            compressed.write(payload)
    return "saved"


def run(days_back: int, root: Path = ROOT) -> dict:
    if days_back < 1 or days_back > 30:
        raise ValueError("days-back must be 1..30")
    now = datetime.now(ZoneInfo("America/Chicago")).date()
    dates = {(now - timedelta(days=i)).isoformat() for i in range(days_back + 1)}
    games = schedule_games(fetch_json(SCHEDULE_URL), dates)
    if not games:
        raise GameDataError(f"No schedule games for {sorted(dates)}")
    report = {"dates": sorted(dates), "scheduled_regular_season_games": len(games),
              "saved": 0, "unchanged": 0, "not_final": 0, "failed": []}
    for g in games:
        gid = g["game_id"]
        boxurl, pbpurl = endpoints(gid)
        try:
            box = fetch_json(boxurl)
            if (box.get("game") or {}).get("gameStatus") != 3:
                report["not_final"] += 1
                continue
            play = fetch_json(pbpurl)
            result = store_final(root, gid, g["game_date"], box, play)
            report[result] += 1
        except (GameDataError, OSError, ValueError, KeyError) as exc:
            report["failed"].append({"game_id": gid, "error": str(exc)[:400]})
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--days-back", type=int, default=3)
    p.add_argument("--root", type=Path, default=ROOT)
    args = p.parse_args()
    report = run(args.days_back, args.root)
    print(json.dumps(report, indent=2), flush=True)
    if report["failed"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
