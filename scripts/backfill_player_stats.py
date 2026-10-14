"""Download a full season of player box scores with resumable page checkpoints."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from v13.player_history import fetch_pages


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--season", type=int, default=2025)
    p.add_argument("--max-pages", type=int, default=600)
    p.add_argument("--out", default="research_data/player_history")
    args = p.parse_args()
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    file = root / f"player_stats_{args.season}.jsonl"
    count = 0
    pages = 0
    more = False
    with file.open("w", encoding="utf-8") as output:
        for page, rows, cursor in fetch_pages(args.season, args.max_pages):
            for row in rows:
                output.write(json.dumps(row, separators=(",", ":")) + "\n")
            count += len(rows)
            pages = page
            more = cursor is not None
            if page % 25 == 0 or not more:
                print(f"season={args.season} pages={page} rows={count}", flush=True)
                output.flush()
    manifest = {
        "season": args.season, "pages": pages, "rows": count,
        "complete": not more, "truncated": more,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "BALLDONTLIE /v1/stats",
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)
    if more:
        raise RuntimeError("Page cap reached; dataset incomplete. Do not use for training.")
    if count < 10000:
        raise RuntimeError(f"Unexpectedly small player-stat dataset: {count}")


if __name__ == "__main__":
    main()
