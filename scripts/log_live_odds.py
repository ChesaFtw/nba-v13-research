"""Append-only snapshot logger for current NBA game odds and player props (closing lines / CLV).

Each run fetches today's and tomorrow's games (America/Chicago), then every vendor's current
game odds and player props, stamps each row with retrieved_at_utc, and writes one JSONL file
per run. The workflow uploads that file to a weekly GitHub Release (under the 1,000-asset cap) so data is never lost to
artifact retention. Never edits or deletes earlier snapshots.

python -m scripts.log_live_odds --out-dir research_data/live_odds
"""
import argparse, json, os, time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

BASE = "https://api.balldontlie.io"


def get(session, path, params, key, tries=6):
    for attempt in range(tries):
        r = session.get(BASE + path, params=params, headers={"Authorization": key}, timeout=40)
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(min(5 * (attempt + 1), 30)); continue
        r.raise_for_status()
        return r.json()
    r.raise_for_status()


def paged(session, path, params, key):
    cursor, rows = None, []
    while True:
        q = list(params) + [("per_page", "100")] + ([("cursor", str(cursor))] if cursor is not None else [])
        payload = get(session, path, q, key)
        rows += payload.get("data", [])
        cursor = (payload.get("meta") or {}).get("next_cursor")
        if cursor is None:
            return rows


def snapshot(key, now=None):
    now = now or datetime.now(timezone.utc)
    local = now.astimezone(ZoneInfo("America/Chicago")).date()
    dates = [local.isoformat(), (local + timedelta(days=1)).isoformat()]
    s = requests.Session()
    games = paged(s, "/v1/games", [("dates[]", d) for d in dates], key)
    upcoming = [g for g in games if g.get("status_state", "scheduled") in ("scheduled", "delayed")
                or str(g.get("status", "")).endswith("ET")]
    stamp = now.isoformat()
    rows = []
    ids = [g["id"] for g in upcoming]
    for i in range(0, len(ids), 10):
        for r in paged(s, "/nba/v2/odds", [("game_ids[]", str(x)) for x in ids[i:i + 10]], key):
            rows.append({"kind": "game_odds", "retrieved_at_utc": stamp, **r})
    for gid in ids:
        payload = get(s, "/nba/v2/odds/player_props", [("game_id", str(gid))], key)
        for r in payload.get("data", []):
            rows.append({"kind": "player_prop", "retrieved_at_utc": stamp, **r})
        time.sleep(0.1)
    tips = {str(g["id"]): g.get("datetime") for g in upcoming}
    for r in rows:
        r["tip_utc"] = tips.get(str(r.get("game_id")))
    return rows, len(upcoming)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="research_data/live_odds")
    a = p.parse_args()
    key = os.environ.get("BALLDONTLIE_API_KEY")
    if not key:
        raise RuntimeError("BALLDONTLIE_API_KEY not set")
    rows, n_games = snapshot(key)
    now = datetime.now(timezone.utc)
    out = Path(a.out_dir) / now.strftime("%Y-%m") / f"snapshot_{now.strftime('%Y%m%dT%H%M%SZ')}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in rows))
    print(json.dumps({"file": str(out), "upcoming_games": n_games, "rows": len(rows),
                      "game_odds": sum(r["kind"] == "game_odds" for r in rows),
                      "player_props": sum(r["kind"] == "player_prop" for r in rows)}))
    gh = os.environ.get("GITHUB_OUTPUT")
    if gh:
        with open(gh, "a") as f:
            f.write(f"file={out}\nweek={now.strftime('%G-W%V')}\nrows={len(rows)}\n")


if __name__ == "__main__":
    main()
