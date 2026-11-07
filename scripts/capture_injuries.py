"""Prospective injury-status capture (2026-27 onward): the news-timing dataset we cannot rebuild later.

Historical official injury PDFs could not be retrieved (R12 probe), and BALLDONTLIE's
/nba/v1/player_injuries is a CURRENT snapshot with no publication timestamp. The only honest way to
know WHEN a status was knowable is to capture it ourselves on a schedule and stamp it with our clock.

Each run writes (unique, timestamped names; never overwritten):
  snapshot.jsonl   one row per listed player: captured_at_utc, player_id, team_id, status,
                   return_date, description, row_sha256
  manifest.json    captured_at_utc, rows, pages, response hashes, set_sha256 (hash of the sorted
                   normalized rows, so unchanged snapshots are recognisable)
status_changes() turns a sequence of snapshots into events with first_seen_utc: the earliest time
WE could have known the new status. Those are the only injury timestamps a backtest may use.
"""
import argparse, hashlib, json, os, sys, time
from datetime import datetime, timezone
from pathlib import Path

URL = "https://api.balldontlie.io/nba/v1/player_injuries"


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def normalize(row, captured):
    pl = row.get("player") or {}
    team = pl.get("team_id") or (row.get("team") or {}).get("id")
    out = {"captured_at_utc": captured, "player_id": pl.get("id"),
           "player_name": " ".join(x for x in (pl.get("first_name"), pl.get("last_name")) if x),
           "team_id": team, "status": row.get("status"), "return_date": row.get("return_date"),
           "description": row.get("description")}
    out["row_sha256"] = sha(json.dumps({k: v for k, v in out.items() if k != "captured_at_utc"}, sort_keys=True))
    return out


def status_changes(snapshots):
    """snapshots: list of (captured_at_utc, [normalized rows]) in any order.
    Returns events {player_id, old_status, new_status, first_seen_utc, previous_seen_utc} whenever a
    player's status differs from his status in the previous snapshot (absent = 'not_listed')."""
    events, last = [], {}
    prev_time = None
    for captured, rows in sorted(snapshots, key=lambda s: s[0]):
        now = {r["player_id"]: r.get("status") or "unknown" for r in rows if r.get("player_id") is not None}
        if prev_time is not None:
            for pid in set(now) | set(last):
                old, new = last.get(pid, "not_listed"), now.get(pid, "not_listed")
                if old != new:
                    events.append({"player_id": pid, "old_status": old, "new_status": new,
                                   "first_seen_utc": captured, "previous_seen_utc": prev_time})
        last, prev_time = now, captured
    return events


def main():
    import requests
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True)
    p.add_argument("--max-pages", type=int, default=30)
    a = p.parse_args()
    key = os.environ.get("BALLDONTLIE_API_KEY")
    if not key:
        print("BALLDONTLIE_API_KEY missing", file=sys.stderr); sys.exit(2)
    now = datetime.now(timezone.utc)
    captured = now.isoformat()
    stem = now.strftime("%Y%m%dT%H%M%SZ")
    out = Path(a.out_dir) / stem
    out.mkdir(parents=True, exist_ok=True)
    s = requests.Session()
    rows, hashes, cursor, pages = [], [], None, 0
    for _ in range(a.max_pages):
        params = {"per_page": 100}
        if cursor is not None:
            params["cursor"] = cursor
        for attempt in range(6):
            r = s.get(URL, headers={"Authorization": key}, params=params, timeout=40)
            if r.status_code in (429, 500, 502, 503, 504) and attempt < 5:
                time.sleep(min(60, 5 * (attempt + 1))); continue
            break
        if r.status_code in (401, 403):
            print(f"key rejected: HTTP {r.status_code}", file=sys.stderr); sys.exit(4)
        r.raise_for_status()
        hashes.append(sha(r.text)); pages += 1
        body = r.json()
        rows += [normalize(x, captured) for x in body.get("data", [])]
        cursor = (body.get("meta") or {}).get("next_cursor")
        if cursor is None:
            break
        time.sleep(0.4)
    with (out / "snapshot.jsonl").open("w") as f:
        for x in rows:
            f.write(json.dumps(x, separators=(",", ":")) + "\n")
    set_hash = sha(json.dumps(sorted(x["row_sha256"] for x in rows)))
    manifest = {"captured_at_utc": captured, "rows": len(rows), "pages": pages, "complete": cursor is None,
                "response_sha256": hashes, "set_sha256": set_hash, "source": URL}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    gh = os.environ.get("GITHUB_OUTPUT")
    if gh:
        with open(gh, "a") as f:
            f.write(f"dir={out}\nstem={stem}\nrows={len(rows)}\nweek={now.strftime('%G-W%V')}\n")


if __name__ == "__main__":
    main()
