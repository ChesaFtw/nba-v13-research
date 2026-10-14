"""Run bounded pilot of historical player-stat availability and save source rows."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from v13.player_history import fetch_pages


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--season", type=int, default=2025)
    p.add_argument("--max-pages", type=int, default=10)
    p.add_argument("--out", default="research_data/player_pilot")
    args = p.parse_args()
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    count = 0
    pages = 0
    has_more = False
    with (root / f"player_stats_{args.season}_pilot.jsonl").open("w") as output:
        for page, rows, cursor in fetch_pages(args.season, args.max_pages):
            for row in rows:
                output.write(json.dumps(row, separators=(",", ":")) + "\n")
            count += len(rows)
            pages = page
            has_more = cursor is not None
            print(f"page={page} rows={len(rows)} total={count}", flush=True)
    if not count:
        raise RuntimeError("No player stats returned")
    report = {"season": args.season, "rows": count, "pages": pages,
              "more_pages_available": has_more,
              "sample_only": has_more,
              "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
              "source": "BALLDONTLIE /v1/stats",
              "warning": "Pilot sample only. Do not treat as complete season or use for model validation."}
    (root / "pilot_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
