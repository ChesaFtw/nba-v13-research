"""Odds-timestamp forensics: what does each historical odds timestamp actually mean?

Samples games per season (stratified by month), pulls every vendor row from BOTH
  /nba/v2/odds/opening      (field: opened_at — documented as "Opening odds timestamp")
  /nba/v2/odds              (field: updated_at — documented as "Last update timestamp")
and writes one row per (game, vendor, endpoint) with tip time, the timestamp, its offset
from tip, and a mechanical classification. It never repairs or reinterprets timestamps.

Classification of offset = timestamp - tip (hours), checked in this order:
  missing_or_unparseable   no usable timestamp / no tip time
  pregame                  offset < 0 (eligible for a pre-tip backtest)
  in_game_window           0 <= offset <= 3       (consistent with live/in-play capture)
  exact_hour_shift         offset in (3, 12] and within 2 min of a whole hour
                           (consistent with a timezone error; flagged, NOT corrected)
  same_or_next_day         3 < offset <= 36       (post-game update or late capture)
  days_or_more_after       offset > 36            (consistent with ingestion/backfill time)

Documented coverage (BALLDONTLIE OpenAPI, opening endpoints): "limited to the most recently
completed season and ongoing seasons where available". Rows outside that are flagged.

python -m scripts.odds_forensics --season 2023 --games games_2023.jsonl --season 2024 --games ... \
  --season 2025 --games ... --per-season 30 --documented-seasons 2025,2026 --out research_results/forensics
"""
import argparse, csv, json, os, random, time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import requests

from v13.pricing import parse_utc

BASE = "https://api.balldontlie.io"
ENDPOINTS = {"opening": ("/nba/v2/odds/opening", "opened_at"), "current": ("/nba/v2/odds", "updated_at")}


def classify(offset_h):
    if offset_h is None:
        return "missing_or_unparseable"
    if offset_h < 0:
        return "pregame"
    if offset_h <= 3:
        return "in_game_window"
    if offset_h <= 12 and abs(offset_h - round(offset_h)) * 60 <= 2:
        return "exact_hour_shift"
    if offset_h <= 36:
        return "same_or_next_day"
    return "days_or_more_after"


def sample_games(games, n, seed=7):
    """Up to n final regular-season games, spread evenly across months (deterministic)."""
    games = [g for g in games if g.get("postseason") is False and g.get("status") == "Final"]
    by_month = defaultdict(list)
    for g in sorted(games, key=lambda g: (str(g["date"]), g["id"])):
        by_month[str(g["date"])[:7]].append(g)
    rng = random.Random(seed)
    months = sorted(by_month)
    out = []
    for i in range(n):
        bucket = by_month[months[i % len(months)]]
        out.append(bucket[rng.randrange(len(bucket))])
    seen, uniq = set(), []
    for g in out:
        if g["id"] not in seen:
            seen.add(g["id"]); uniq.append(g)
    return uniq


def fetch(session, key, path, game_id):
    for attempt in range(6):
        r = session.get(BASE + path, params=[("game_ids[]", str(game_id)), ("per_page", "100")],
                        headers={"Authorization": key}, timeout=40)
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(5 * (attempt + 1)); continue
        if r.status_code in (400, 404):
            return [], f"http_{r.status_code}"
        r.raise_for_status()
        return r.json().get("data", []), "ok"
    return [], "http_retry_exhausted"


def rows_for_game(g, season, fetched, documented):
    tip = parse_utc(g.get("datetime"))
    out = []
    for endpoint, (data, status) in fetched.items():
        field = ENDPOINTS[endpoint][1]
        if not data:
            out.append({"season": season, "game_id": g["id"], "game_date": g["date"], "tip_utc": g.get("datetime"),
                        "endpoint": endpoint, "vendor": None, "spread_home_value": None, "total_value": None,
                        "timestamp_field": field, "timestamp": None, "offset_hours": None,
                        "classification": f"no_rows_{status}", "pregame_eligible": False,
                        "in_documented_coverage": season in documented})
            continue
        for r in data:
            ts = parse_utc(r.get(field))
            off = round((ts - tip).total_seconds() / 3600, 3) if ts and tip else None
            cls = classify(off)
            out.append({"season": season, "game_id": g["id"], "game_date": g["date"], "tip_utc": g.get("datetime"),
                        "endpoint": endpoint, "vendor": r.get("vendor"), "spread_home_value": r.get("spread_home_value"),
                        "total_value": r.get("total_value"), "timestamp_field": field, "timestamp": r.get(field),
                        "offset_hours": off, "classification": cls,
                        "pregame_eligible": cls == "pregame" and endpoint == "opening" and season in documented,
                        "in_documented_coverage": season in documented})
    return out


def summarize(rows):
    out = {}
    for (season, endpoint) in sorted({(r["season"], r["endpoint"]) for r in rows}):
        sub = [r for r in rows if r["season"] == season and r["endpoint"] == endpoint]
        offs = [r["offset_hours"] for r in sub if r["offset_hours"] is not None]
        whole = [o for o in offs if o > 0 and abs(o - round(o)) * 60 <= 2]
        out[f"{season}|{endpoint}"] = {
            "rows": len(sub), "games": len({r["game_id"] for r in sub}),
            "classification": dict(Counter(r["classification"] for r in sub)),
            "offset_hours_p5_p50_p95": np.percentile(offs, [5, 50, 95]).round(2).tolist() if offs else None,
            "share_positive_offsets_on_whole_hours": round(len(whole) / max(1, sum(o > 0 for o in offs)), 3),
            "pregame_eligible_rows": sum(r["pregame_eligible"] for r in sub),
            "in_documented_coverage": sub[0]["in_documented_coverage"] if sub else None}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--season", type=int, action="append", required=True)
    p.add_argument("--games", action="append", required=True)
    p.add_argument("--per-season", type=int, default=30)
    p.add_argument("--documented-seasons", default="2025,2026")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    key = os.environ.get("BALLDONTLIE_API_KEY")
    if not key:
        raise RuntimeError("BALLDONTLIE_API_KEY not set")
    documented = {int(x) for x in a.documented_seasons.split(",") if x}
    s = requests.Session(); rows = []
    for season, path in zip(a.season, a.games):
        games = [json.loads(x) for x in Path(path).open() if x.strip()]
        for g in sample_games(games, a.per_season):
            fetched = {ep: fetch(s, key, ENDPOINTS[ep][0], g["id"]) for ep in ENDPOINTS}
            rows += rows_for_game(g, season, fetched, documented)
            time.sleep(0.15)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    with (out / "odds_forensics_rows.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    summary = {"documented_seasons": sorted(documented),
               "source_note": "BALLDONTLIE OpenAPI: opening endpoints cover 'the most recently completed season and "
                              "ongoing seasons where available'; prop opened_at is described as 'last updated'.",
               "by_season_endpoint": summarize(rows)}
    (out / "odds_forensics_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
