"""Round 13 data inventory: what can each feature family actually be built from?

--api   bounded probe of BALLDONTLIE endpoints (about 30 requests). Per season it records HTTP status,
        rows, every field name, and the share of non-null values for each field. That decides whether
        the tracking / hustle / matchup fields in /nba/v2/stats/advanced (rebound chances, touches,
        passes, contested shots, box outs, matchup_*) exist historically or only recently.
--props local inventory of the saved opening-prop file: prop types, market types, vendors, and how many
        rows carry two-sided prices. That decides which prop markets can be evaluated at all.
Nothing is modelled here.
"""
import argparse, json, os, time
from collections import Counter, defaultdict
from pathlib import Path

API = "https://api.balldontlie.io"
SEASON_DATES = {2023: ("2023-11-15", "2024-01-17", "2024-03-20"), 2024: ("2024-11-13", "2025-01-15", "2025-03-19"),
                2025: ("2025-11-12", "2026-01-14", "2026-03-18")}
KEY_FIELDS = ("rebound_chances_total", "rebound_chances_off", "rebound_chances_def", "box_outs", "touches", "passes",
              "secondary_assists", "free_throw_assists", "contested_fga", "uncontested_fga", "defended_at_rim_fga",
              "matchup_minutes", "matchup_player_points", "switches_on", "speed", "distance", "deflections",
              "usage_percentage", "pace", "pct_fga_3pt", "pct_pts_paint", "fouls_drawn", "points_second_chance")


def field_coverage(rows):
    """{field: share of rows where the value is not None} over the union of fields (nested dicts skipped)."""
    if not rows:
        return {}
    counts, n = Counter(), len(rows)
    fields = set()
    for r in rows:
        for k, v in r.items():
            if isinstance(v, dict):
                continue
            fields.add(k)
            if v is not None:
                counts[k] += 1
    return {k: round(counts[k] / n, 3) for k in sorted(fields)}


def prop_inventory(props):
    by_type = defaultdict(Counter)
    vendors = Counter()
    for r in props:
        t = str(r.get("prop_type"))
        m = r.get("market") or {}
        by_type[t]["rows"] += 1
        by_type[t][f"market_{m.get('type')}"] += 1
        if m.get("type") == "over_under" and m.get("over_odds") is not None and m.get("under_odds") is not None:
            by_type[t]["two_sided"] += 1
        vendors[str(r.get("vendor"))] += 1
    games = len({r.get("game_id") for r in props})
    return {"rows": len(props), "games": games, "vendors": dict(vendors),
            "prop_types": {k: dict(v) for k, v in sorted(by_type.items(), key=lambda kv: -kv[1]["rows"])}}


def _get(session, url, params, key, tries=5):
    for attempt in range(tries):
        try:
            r = session.get(url, headers={"Authorization": key}, params=params, timeout=40)
        except Exception as e:  # network failure is a finding, not a crash
            if attempt == tries - 1:
                return {"error": f"{type(e).__name__}: {e}"}
            time.sleep(5 * (attempt + 1)); continue
        if r.status_code in (429, 500, 502, 503, 504) and attempt < tries - 1:
            time.sleep(min(60, 5 * (attempt + 1))); continue
        try:
            body = r.json()
        except ValueError:
            body = {"text": r.text[:300]}
        return {"status": r.status_code, "body": body}
    return {"error": "retries exhausted"}


def summarize(resp, period_zero=False):
    if "error" in resp:
        return {"error": resp["error"]}
    body = resp["body"] if isinstance(resp.get("body"), dict) else {}
    rows = body.get("data") if isinstance(body.get("data"), list) else []
    if period_zero:
        whole = [r for r in rows if r.get("period") in (0, None)]
    else:
        whole = rows
    cov = field_coverage(whole)
    return {"status": resp["status"], "rows": len(rows), "rows_used": len(whole),
            "periods": dict(Counter(str(r.get("period")) for r in rows)) if rows and "period" in rows[0] else None,
            "key_field_coverage": {k: cov.get(k) for k in KEY_FIELDS if k in cov} or None,
            "all_field_coverage": cov, "message": None if rows else str(body)[:300]}


def api_probe(out):
    import requests
    key = os.environ.get("BALLDONTLIE_API_KEY")
    if not key:
        raise RuntimeError("BALLDONTLIE_API_KEY missing")
    s = requests.Session()
    res = {"advanced_v2": {}, "advanced_v1": {}, "plays": {}, "lineups": {}, "injuries_snapshot": None}
    for season, dates in SEASON_DATES.items():
        for d in dates:
            res["advanced_v2"][f"{season}|{d}"] = summarize(
                _get(s, f"{API}/nba/v2/stats/advanced", {"seasons[]": season, "dates[]": d, "per_page": 100}, key), True)
            time.sleep(0.5)
        res["advanced_v1"][str(season)] = summarize(
            _get(s, f"{API}/nba/v1/stats/advanced", {"seasons[]": season, "dates[]": dates[1], "per_page": 100}, key), True)
        g = _get(s, f"{API}/v1/games", {"dates[]": dates[1], "per_page": 5}, key)
        gid = None
        if "body" in g and isinstance(g["body"], dict) and g["body"].get("data"):
            gid = g["body"]["data"][0]["id"]
        if gid is None:
            res["plays"][str(season)] = res["lineups"][str(season)] = {"error": "no game id found"}
            continue
        for name, params in (("game_id", {"game_id": gid}), ("game_ids[]", {"game_ids[]": gid, "per_page": 100})):
            sm = summarize(_get(s, f"{API}/nba/v1/plays", params, key))
            sm.pop("all_field_coverage", None)
            res["plays"][f"{season}|{name}"] = {**sm, "game_id": gid}
            time.sleep(0.5)
        sm = summarize(_get(s, f"{API}/nba/v1/lineups", {"game_ids[]": gid, "per_page": 100}, key))
        res["lineups"][str(season)] = {**sm, "game_id": gid}
    res["injuries_snapshot"] = summarize(_get(s, f"{API}/nba/v1/player_injuries", {"per_page": 100}, key))
    verdict = {}
    for season in SEASON_DATES:
        cov = [v.get("key_field_coverage") or {} for k, v in res["advanced_v2"].items() if k.startswith(str(season))]
        rc = [c.get("rebound_chances_total") for c in cov if c.get("rebound_chances_total") is not None]
        tc = [c.get("touches") for c in cov if c.get("touches") is not None]
        verdict[str(season)] = {"tracking_rebound_chances_available": bool(rc) and min(rc) >= 0.8,
                                "tracking_touches_available": bool(tc) and min(tc) >= 0.8}
    res["tracking_verdict_by_season"] = verdict
    out.mkdir(parents=True, exist_ok=True)
    (out / "api_inventory.json").write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps({"tracking_verdict_by_season": verdict,
                      "advanced_v2": {k: (v.get("status"), v.get("rows"), v.get("key_field_coverage")) for k, v in res["advanced_v2"].items()},
                      "plays": {k: (v.get("status"), v.get("rows")) for k, v in res["plays"].items()},
                      "lineups": {k: (v.get("status"), v.get("rows")) for k, v in res["lineups"].items()}}, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--api", action="store_true")
    p.add_argument("--props")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    out = Path(a.out)
    if a.props:
        rows = [json.loads(x) for x in Path(a.props).open() if x.strip()]
        inv = prop_inventory(rows)
        out.mkdir(parents=True, exist_ok=True)
        (out / "prop_market_inventory.json").write_text(json.dumps(inv, indent=2) + "\n")
        print(json.dumps(inv, indent=2))
    if a.api:
        api_probe(out)


if __name__ == "__main__":
    main()
