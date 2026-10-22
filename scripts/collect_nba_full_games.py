"""Collect every completed regular-season NBA game from accessible ESPN public feeds.

Archive includes full PBP and complete box-score JSON, plus normalized features.
Sources are versioned by SHA256. Incomplete or mismatched games fail visibly.
"""
import argparse
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from pathlib import Path
import gzip
import hashlib
import json
import sys

from v13.espn_game_ingest import (
    ESPNDataError, get_json, scoreboard_url, summary_url,
    regular_final_events, normalize,
)

ROOT = Path("research_data/live_nba")


def _bytes(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def store_final(root: Path, event: dict, summary: dict, day: str):
    game_id = str(event["id"])
    normalized = normalize(event, summary, day)
    raw = {"provider": "ESPN public NBA", "scoreboard_event": event,
           "full_summary": summary}
    payload = _bytes(raw)
    digest = hashlib.sha256(payload).hexdigest()
    directory = root / day
    directory.mkdir(parents=True, exist_ok=True)
    metadata_path = directory / f"{game_id}.json"
    source_path = directory / f"{game_id}.raw.json.gz"
    if metadata_path.exists() and source_path.exists():
        prior = json.loads(metadata_path.read_text(encoding="utf-8"))
        if prior.get("raw_sha256") == digest:
            return "unchanged"
    normalized["raw_sha256"] = digest
    metadata_path.write_bytes(_bytes(normalized) + b"\n")
    with source_path.open("wb") as out:
        with gzip.GzipFile(fileobj=out, mode="wb", mtime=0) as zipped:
            zipped.write(payload)
    return "saved"


def run(days_back: int, root: Path = ROOT) -> dict:
    if days_back < 1 or days_back > 30:
        raise ValueError("days-back must be between 1 and 30")
    today = datetime.now(ZoneInfo("America/Chicago")).date()
    days = [(today - timedelta(days=i)).isoformat()
            for i in range(days_back, -1, -1)]
    report = {"dates": days, "scheduled_reg_season": 0, "completed_reg_season": 0,
              "saved": 0, "unchanged": 0, "failed": []}
    for day in days:
        board = get_json(scoreboard_url(day))
        regular = [e for e in board.get("events") or []
                   if (e.get("season") or {}).get("type") == 2]
        report["scheduled_reg_season"] += len(regular)
        for event in regular_final_events(board):
            eid = str(event["id"])
            report["completed_reg_season"] += 1
            try:
                summary = get_json(summary_url(eid))
                result = store_final(root, event, summary, day)
                report[result] += 1
            except (ESPNDataError, OSError, ValueError, KeyError, TypeError) as exc:
                report["failed"].append({"game_id": eid, "error": str(exc)[:400]})
    if report["scheduled_reg_season"] == 0:
        raise ESPNDataError(f"Zero regular season games discovered on {days}")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days-back", type=int, default=3)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    report = run(args.days_back, args.root)
    print(json.dumps(report, indent=2), flush=True)
    if report["failed"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
