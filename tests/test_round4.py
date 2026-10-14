import random
from datetime import date, timedelta

import scripts.log_live_odds as logger
from scripts.cross_market_consistency import build, run


def consistency_world(seed=0, games=500, stale=1.5):
    rng = random.Random(seed)
    teams = {t: [rng.uniform(6, 26) for _ in range(5)] for t in range(1, 9)}
    G, ODDS, PROPS, STATS = [], [], [], []
    for i in range(games):
        day = (date(2025, 10, 22) + timedelta(days=i // 3)).isoformat()
        h, a = rng.sample(sorted(teams), 2)
        gid = i + 1
        pts = {}
        for team in (h, a):
            shade = rng.choice((-stale, 0.0, stale))  # the props desk is stale for this team-game
            total = 0
            for k, mu in enumerate(teams[team]):
                pid = team * 100 + k
                x = max(0, round(rng.gauss(mu, 5)))
                total += x
                STATS.append({"pts": x, "player": {"id": pid}, "team": {"id": team}, "game": {"id": gid, "date": day}})
                PROPS.append({"id": len(PROPS), "game_id": gid, "player_id": pid, "vendor": "caesars", "prop_type": "points",
                              "line_value": str(round(mu + shade) + 0.5), "opened_at": day + "T15:00:00Z",
                              "market": {"type": "over_under", "over_odds": -110, "under_odds": -110}})
            pts[team] = total
        exp_h, exp_a = sum(teams[h]), sum(teams[a])
        G.append({"id": gid, "date": day, "datetime": day + "T23:00:00Z", "home_team": {"id": h}, "visitor_team": {"id": a},
                  "home_team_score": pts[h], "visitor_team_score": pts[a], "status": "Final"})
        ODDS.append({"id": gid, "game_id": gid, "vendor": "draftkings", "opened_at": day + "T14:00:00Z",
                     "spread_home_value": str(-(exp_h - exp_a)), "total_value": str(exp_h + exp_a)})
    # warm-up rows so players have a team before their first scored game
    for team, mus in teams.items():
        for k in range(5):
            STATS.append({"pts": 0, "player": {"id": team * 100 + k}, "team": {"id": team}, "game": {"id": -team, "date": "2025-10-01"}})
    return G, ODDS, PROPS, STATS


def test_consistency_detects_stale_props():
    rows = build(*consistency_world())
    res = run(rows, min_train=100)
    assert res["team_games_scored"] > 100
    assert res["corr_gap_with_player_points_minus_lines"] > 0.2
    assert res["rule_bets"] > 50 and res["rule_roi_flat_pct"] > 0


def test_consistency_no_edge_when_props_are_right():
    res = run(build(*consistency_world(seed=1, stale=0.0)), min_train=100)
    assert res["rule_roi_flat_pct"] is None or res["rule_roi_flat_pct"] < 8


def test_live_logger_stamps_rows():
    class Resp:
        status_code = 200
        def __init__(self, payload): self.payload = payload
        def json(self): return self.payload
        def raise_for_status(self): pass
    class Session:
        def get(self, url, params=None, headers=None, timeout=None):
            if url.endswith("/v1/games"):
                return Resp({"data": [{"id": 5, "status_state": "scheduled", "datetime": "2026-10-21T23:30:00Z"}], "meta": {}})
            if url.endswith("/nba/v2/odds"):
                return Resp({"data": [{"game_id": 5, "vendor": "dk", "total_value": "221.5"}], "meta": {}})
            return Resp({"data": [{"game_id": 5, "player_id": 9, "prop_type": "points"}]})
    original = logger.requests.Session
    logger.requests.Session = Session
    try:
        rows, n = logger.snapshot("k")
    finally:
        logger.requests.Session = original
    assert n == 1 and len(rows) == 2
    assert all(r["retrieved_at_utc"] and r["tip_utc"] == "2026-10-21T23:30:00Z" for r in rows)
