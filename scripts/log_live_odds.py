"""Append-only live odds logger (v2): normalized, provenance-stamped observations.

Each run captures current BALLDONTLIE game odds (spread / total / moneyline) for today's and
tomorrow's games (America/Chicago) and current player props for games tipping within
--props-window-hours. Every quote side becomes one observation row:

  observation_id   sha256(quote_id | captured_at_utc)   unique per capture
  quote_id         sha256(endpoint|game|player|book|market|side|line|odds|vendor_ts)  same quote seen again -> same id
  captured_at_utc  our clock (the time we SAW the quote)        vendor_updated_at  the vendor's timestamp
  game_id, tip_utc, seconds_to_tip, player_id, sportsbook, market, side, line, american_odds
  source_endpoint, response_sha256 (hash of the raw HTTP body), logger_version, git_commit
  stale_vendor_quote (vendor timestamp older than --stale-hours at capture)

Outputs per run (unique, timestamped names; never overwritten): observations.jsonl,
raw_responses.jsonl (endpoint, params, status, sha256, body) and manifest.json.

Fails loudly: missing key (exit 2), rejected key (exit 4), schema drift above --max-schema-error-share
(exit 3). A per-run request budget (--max-requests) stops the run early and is recorded.
"""
import argparse, hashlib, json, os, sys, time
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

from v13.pricing import parse_utc

BASE = "https://api.balldontlie.io"
LOGGER_VERSION = "live-logger-2"
GAME_REQUIRED = ("game_id", "vendor", "updated_at")
PROP_REQUIRED = ("game_id", "player_id", "vendor", "prop_type", "market", "updated_at")


class CredentialError(RuntimeError):
    pass


class Budget:
    def __init__(self, max_requests):
        self.max, self.used, self.exhausted = max_requests, 0, False

    def take(self):
        if self.used >= self.max:
            self.exhausted = True
            return False
        self.used += 1
        return True


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def get(session, path, params, key, budget, raw_log, captured, tries=6):
    """One budgeted request with Retry-After backoff; every raw body is hashed and kept."""
    for attempt in range(tries):
        if not budget.take():
            return None
        r = session.get(BASE + path, params=params, headers={"Authorization": key}, timeout=40)
        body = r.text
        raw_log.append({"captured_at_utc": captured, "endpoint": path, "params": params, "status": r.status_code,
                        "sha256": sha(body), "body": body if r.status_code == 200 else body[:500]})
        if r.status_code in (401, 403):
            raise CredentialError(f"BALLDONTLIE rejected the API key on {path} (HTTP {r.status_code})")
        if r.status_code == 429 or r.status_code >= 500:
            wait = r.headers.get("Retry-After") if hasattr(r, "headers") else None
            try:
                delay = min(max(float(wait), 1), 60) if wait else min(5 * (attempt + 1), 60)
            except ValueError:
                delay = min(5 * (attempt + 1), 60)
            time.sleep(delay)
            continue
        r.raise_for_status()
        payload = r.json()
        payload["_sha256"] = raw_log[-1]["sha256"]
        return payload
    raise RuntimeError(f"{path}: retries exhausted")


def _obs(endpoint, row, market, side, line, odds, captured, tip, sha256, commit, stale_hours):
    vendor_ts = row.get("updated_at")
    quote = "|".join(str(x) for x in (endpoint, row.get("game_id"), row.get("player_id"), row.get("vendor"),
                                      market, side, line, odds, vendor_ts))
    qid = sha(quote)
    cap, vt = parse_utc(captured), parse_utc(vendor_ts)
    return {"observation_id": sha(qid + "|" + captured), "quote_id": qid, "captured_at_utc": captured,
            "vendor_updated_at": vendor_ts, "game_id": row.get("game_id"), "tip_utc": tip.isoformat() if tip else None,
            "seconds_to_tip": int((tip - cap).total_seconds()) if tip and cap else None,
            "player_id": row.get("player_id"), "sportsbook": row.get("vendor"), "market": market, "side": side,
            "line": line, "american_odds": odds, "source_endpoint": endpoint, "response_sha256": sha256,
            "logger_version": LOGGER_VERSION, "git_commit": commit,
            "stale_vendor_quote": bool(cap and vt and cap - vt > timedelta(hours=stale_hours))}


def normalize_game(row, captured, tip, sha256, commit, stale_hours):
    ep = "/nba/v2/odds"
    pairs = [("spread", "home", row.get("spread_home_value"), row.get("spread_home_odds")),
             ("spread", "away", row.get("spread_away_value"), row.get("spread_away_odds")),
             ("total", "over", row.get("total_value"), row.get("total_over_odds")),
             ("total", "under", row.get("total_value"), row.get("total_under_odds")),
             ("moneyline", "home", None, row.get("moneyline_home_odds")),
             ("moneyline", "away", None, row.get("moneyline_away_odds"))]
    return [_obs(ep, row, m, s, l, o, captured, tip, sha256, commit, stale_hours) for m, s, l, o in pairs if o is not None]


def normalize_prop(row, captured, tip, sha256, commit, stale_hours):
    ep, mk = "/nba/v2/odds/player_props", row.get("market") or {}
    market = f"prop_{row.get('prop_type')}"
    if mk.get("type") == "over_under":
        sides = [("over", mk.get("over_odds")), ("under", mk.get("under_odds"))]
    elif mk.get("type") == "milestone":
        sides = [("milestone_yes", mk.get("odds"))]
    else:
        return None  # schema drift
    return [_obs(ep, row, market, s, row.get("line_value"), o, captured, tip, sha256, commit, stale_hours)
            for s, o in sides if o is not None]


def snapshot(key, now=None, max_requests=80, props_window_hours=14, stale_hours=6, commit=None, session=None):
    now = now or datetime.now(timezone.utc)
    captured = now.isoformat()
    s = session or requests.Session()
    budget, raw, schema = Budget(max_requests), [], Counter()
    local = now.astimezone(ZoneInfo("America/Chicago")).date()
    dates = [local.isoformat(), (local + timedelta(days=1)).isoformat()]
    games, cursor = [], None
    while True:
        params = [("dates[]", d) for d in dates] + [("per_page", "100")] + ([("cursor", str(cursor))] if cursor else [])
        payload = get(s, "/v1/games", params, key, budget, raw, captured)
        if payload is None:
            break
        games += payload.get("data", [])
        cursor = (payload.get("meta") or {}).get("next_cursor")
        if not cursor:
            break
    tips = {str(g["id"]): parse_utc(g.get("datetime")) for g in games}
    upcoming = [g for g in games if tips.get(str(g["id"])) and tips[str(g["id"])] > now]
    obs = []
    ids = [g["id"] for g in upcoming]
    for i in range(0, len(ids), 10):
        payload = get(s, "/nba/v2/odds", [("game_ids[]", str(x)) for x in ids[i:i + 10]] + [("per_page", "100")],
                      key, budget, raw, captured)
        if payload is None:
            break
        for r in payload.get("data", []):
            if any(r.get(k) is None for k in GAME_REQUIRED):
                schema["game_odds_missing_required"] += 1; continue
            obs += normalize_game(r, captured, tips.get(str(r["game_id"])), payload["_sha256"], commit, stale_hours)
            schema["game_odds_rows"] += 1
    soon = [g for g in upcoming if tips[str(g["id"])] - now <= timedelta(hours=props_window_hours)]
    for g in soon:
        payload = get(s, "/nba/v2/odds/player_props", [("game_id", str(g["id"]))], key, budget, raw, captured)
        if payload is None:
            break
        for r in payload.get("data", []):
            if any(r.get(k) is None for k in PROP_REQUIRED):
                schema["prop_missing_required"] += 1; continue
            rows = normalize_prop(r, captured, tips.get(str(r["game_id"])), payload["_sha256"], commit, stale_hours)
            if rows is None:
                schema["prop_unknown_market_type"] += 1; continue
            obs += rows; schema["prop_rows"] += 1
        time.sleep(0.1)
    seen, unique = set(), []
    for o in obs:
        if o["observation_id"] not in seen:
            seen.add(o["observation_id"]); unique.append(o)
    bad = schema["game_odds_missing_required"] + schema["prop_missing_required"] + schema["prop_unknown_market_type"]
    total = bad + schema["game_odds_rows"] + schema["prop_rows"]
    manifest = {"captured_at_utc": captured, "logger_version": LOGGER_VERSION, "git_commit": commit,
                "games_found": len(games), "upcoming_games": len(upcoming), "prop_games_polled": len(soon),
                "requests_used": budget.used, "request_budget": budget.max, "budget_exhausted": budget.exhausted,
                "observations": len(unique), "duplicates_dropped": len(obs) - len(unique),
                "stale_vendor_quotes": sum(o["stale_vendor_quote"] for o in unique),
                "schema": dict(schema), "schema_error_share": round(bad / total, 4) if total else 0.0}
    return unique, raw, manifest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="research_data/live_odds")
    p.add_argument("--max-requests", type=int, default=80)
    p.add_argument("--props-window-hours", type=float, default=14)
    p.add_argument("--stale-hours", type=float, default=6)
    p.add_argument("--max-schema-error-share", type=float, default=0.2)
    a = p.parse_args()
    key = os.environ.get("BALLDONTLIE_API_KEY")
    if not key:
        print("ERROR: BALLDONTLIE_API_KEY is not set; nothing captured.", file=sys.stderr)
        sys.exit(2)
    now = datetime.now(timezone.utc)
    try:
        obs, raw, manifest = snapshot(key, now, a.max_requests, a.props_window_hours, a.stale_hours,
                                      os.environ.get("GITHUB_SHA"))
    except CredentialError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(4)
    stem = now.strftime("%Y%m%dT%H%M%SZ")
    out = Path(a.out_dir) / now.strftime("%G-W%V") / stem
    out.mkdir(parents=True, exist_ok=False)  # never reuse a run directory
    (out / "observations.jsonl").write_text("".join(json.dumps(o, separators=(",", ":")) + "\n" for o in obs))
    (out / "raw_responses.jsonl").write_text("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in raw))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    gh = os.environ.get("GITHUB_OUTPUT")
    if gh:
        with open(gh, "a") as f:
            f.write(f"dir={out}\nstem={stem}\nweek={now.strftime('%G-W%V')}\nrows={len(obs)}\n")
    if manifest["schema_error_share"] > a.max_schema_error_share:
        print("ERROR: schema drift above threshold; data kept, run marked failed.", file=sys.stderr)
        sys.exit(3)


if __name__ == "__main__":
    main()
