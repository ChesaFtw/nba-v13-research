"""Pregame player matchup features from box scores only (no new API calls).

Every feature for a player-game uses only games dated strictly before it. Sources:
  player object nested in /v1/stats rows: height ("6-11"), weight, position, draft_year
  team-game aggregates built from the same box scores (FGA, FGM, FG3A, minutes).

Features
  height_in                 player height (inches)
  opp_size_in               minutes-weighted height of the opponent's top-8 minute players over its last 10 games
  height_diff               height_in - opp_size_in
  opp_missed_fg             opponent missed field goals per game, last 10 (defensive-rebound chances)
  own_missed_fg             own team missed field goals per game, last 10 (offensive-rebound chances)
  opp_fg3a_rate             opponent 3PA / FGA, last 10 (long rebounds)
  years_since_draft         season - draft_year (career-stage proxy; BALLDONTLIE has no birth date)
  b2b                       player's previous game was the day before
  h2h_resid_<stat>          shrunk mean of (actual - player's 10-game average) in earlier meetings with this
                            opponent, all seasons supplied; shrinkage k = 5 games
"""
from collections import defaultdict, deque
from datetime import date

from scripts.regular_season_filter import minutes as parse_minutes

H2H_K = 5.0
STATS = {"points": "pts", "rebounds": "reb", "assists": "ast"}


def parse_height(h):
    try:
        ft, inch = str(h).split("-")
        return 12 * int(ft) + int(inch)
    except (ValueError, AttributeError):
        return None


def _f(r, k):
    try:
        return float(r.get(k) or 0)
    except (TypeError, ValueError):
        return 0.0


def build(stats, games):
    """(game_id, player_id) -> feature dict, built strictly from earlier games."""
    gmeta = {str(g["id"]): g for g in games}
    rows = []
    for r in stats:
        g, pl, tm = r.get("game") or {}, r.get("player") or {}, r.get("team") or {}
        if g.get("id") is None or pl.get("id") is None or tm.get("id") is None:
            continue
        gid = str(g["id"]); meta = gmeta.get(gid, {})
        rows.append({"gid": gid, "date": str(g.get("date") or meta.get("date"))[:10], "season": meta.get("season", g.get("season")),
                     "pid": str(pl["id"]), "tid": str(tm["id"]), "min": parse_minutes(r.get("min")), "row": r, "pl": pl})
    rows.sort(key=lambda x: (x["date"], x["gid"]))
    # opponent per (game, team)
    teams_in_game = defaultdict(set)
    for x in rows:
        teams_in_game[x["gid"]].add(x["tid"])
    opp = {(gid, t): next(iter(ts - {t}), None) for gid, ts in teams_in_game.items() for t in ts}

    team_hist = defaultdict(lambda: deque(maxlen=10))     # tid -> recent team-game aggregates
    player_hist = defaultdict(lambda: deque(maxlen=10))   # pid -> recent stat dicts
    h2h = defaultdict(list)                               # (pid, opp_tid, stat) -> residuals
    last_date = {}
    out = {}
    by_date = defaultdict(list)
    for x in rows:
        by_date[x["date"]].append(x)
    for day in sorted(by_date):
        today = by_date[day]
        # 1) features from strictly earlier days
        for x in today:
            o = opp.get((x["gid"], x["tid"]))
            th, oh = list(team_hist[x["tid"]]), list(team_hist[o]) if o else []
            hgt = parse_height(x["pl"].get("height"))
            feats = {"height_in": hgt, "years_since_draft": None, "b2b": None}
            dy = x["pl"].get("draft_year")
            if dy and x["season"]:
                feats["years_since_draft"] = int(x["season"]) - int(dy)
            if x["pid"] in last_date:
                feats["b2b"] = int((date.fromisoformat(day) - date.fromisoformat(last_date[x["pid"]])).days == 1)
            if len(oh) >= 5:
                feats["opp_missed_fg"] = sum(t["fga"] - t["fgm"] for t in oh) / len(oh)
                feats["opp_fg3a_rate"] = sum(t["fg3a"] for t in oh) / max(1.0, sum(t["fga"] for t in oh))
                sz = [t["size"] for t in oh if t["size"] is not None]
                feats["opp_size_in"] = sum(sz) / len(sz) if sz else None
                if hgt is not None and feats["opp_size_in"] is not None:
                    feats["height_diff"] = hgt - feats["opp_size_in"]
            if len(th) >= 5:
                feats["own_missed_fg"] = sum(t["fga"] - t["fgm"] for t in th) / len(th)
            ph = list(player_hist[x["pid"]])
            for name, key in STATS.items():
                res = h2h.get((x["pid"], o, name), [])
                feats[f"h2h_resid_{name}"] = sum(res) / (len(res) + H2H_K) if o else None
                feats[f"avg10_{name}"] = sum(p[key] for p in ph) / len(ph) if ph else None
            out[(x["gid"], x["pid"])] = feats
        # 2) then update histories with today's results
        agg = defaultdict(lambda: {"fga": 0.0, "fgm": 0.0, "fg3a": 0.0, "players": []})
        for x in today:
            a = agg[(x["gid"], x["tid"])]
            for k in ("fga", "fgm", "fg3a"):
                a[k] += _f(x["row"], k)
            a["players"].append((x["min"], parse_height(x["pl"].get("height"))))
        for (gid, tid), a in agg.items():
            top = sorted((p for p in a["players"] if p[1] is not None), reverse=True)[:8]
            mins = sum(m for m, _ in top)
            a["size"] = sum(m * h for m, h in top) / mins if mins else None
            team_hist[tid].append(a)
        for x in today:
            o = opp.get((x["gid"], x["tid"]))
            ph = list(player_hist[x["pid"]])
            vals = {k: _f(x["row"], k) for k in STATS.values()}
            if len(ph) >= 5 and o:
                for name, key in STATS.items():
                    h2h[(x["pid"], o, name)].append(vals[key] - sum(p[key] for p in ph) / len(ph))
            player_hist[x["pid"]].append(vals)
            last_date[x["pid"]] = day
    for f in out.values():
        if f.get("height_diff") is not None and f.get("opp_missed_fg") is not None:
            f["height_x_oppmiss"] = f["height_diff"] * f["opp_missed_fg"]
        if f.get("years_since_draft") is not None and f.get("b2b") is not None:
            f["yrs_x_b2b"] = f["years_since_draft"] * f["b2b"]
    return out
