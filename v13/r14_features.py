"""Round 14 pregame features for three-pointers and rebounds (box scores + optional tracking).

Every value for a player-game uses only games on strictly earlier days, and the day's results
are folded into history only after all of that day's features exist (same rule as R11/R13).

Three-pointers (attempts separated from conversion)
  fg3a_pm            EWMA (half-life 10 games) of 3PA per minute, games with >= 5 min
  fg3_pct_shrunk     (career-to-date 3PM + K3 * league%) / (career-to-date 3PA + K3), league% to date
  hot_gap            last-5-game 3P% minus fg3_pct_shrunk (None if < 10 attempts in those games)
  attempt_trend      3PA/min over last 5 games minus last 20
  vacated_ast        last-10 assists per game of rotation teammates absent from the team's previous game
Rebounds (team opportunities -> player share)
  exp_dreb_opps      0.5 * opponent's own missed FG (L10) + 0.5 * misses our defence forced (L10)
  exp_oreb_opps      0.5 * our missed FG (L10) + 0.5 * misses the opponent's defence forced (L10)
  dreb_share         EWMA of DREB / (opponent misses x min/48), shrunk to the position mean to date
  oreb_share         EWMA of OREB / (own misses x min/48), shrunk likewise
  opp_oreb_rate      opponent OREB / opponent missed FG (L10): competition for our defensive boards
  vacated_reb        last-10 rebounds per game of rotation teammates absent from the previous game
  height_diff, opp_fg3a_rate come from v13.matchup_features (R11)
Tracking (only when supplied and covered; /nba/v2/stats/advanced period-0 rows)
  chances_pm         EWMA of rebound_chances_total per minute
  chance_conv        (career-to-date REB + KC * league conversion) / (chances + KC)
"""
from collections import defaultdict, deque

from scripts.regular_season_filter import minutes as parse_minutes
from v13.matchup_features import build as build_matchup
from v13.minutes_model import ROTATION_MIN, ewma

HL = 10.0
K3 = 150.0     # shrinkage (attempts) for 3P%
KS = 5.0       # shrinkage (games) for rebound shares toward the position mean
KC = 30.0      # shrinkage (chances) for rebound conversion


def _f(r, k):
    try:
        return float(r.get(k) or 0)
    except (TypeError, ValueError):
        return 0.0


def pos_group(p):
    p = str(p or "").upper()
    return "C" if p.startswith("C") else ("F" if p.startswith("F") else ("G" if p.startswith("G") else "U"))


def build(stats, games, tracking=None):
    """(gid, pid) -> feature dict. `tracking`: optional {(gid, pid): period-0 advanced row}."""
    matchup = build_matchup(stats, games)
    gmeta = {str(g["id"]): g for g in games}
    team_sched = defaultdict(list)
    for g in games:
        if g.get("postseason"):
            continue
        for side in ("home_team", "visitor_team"):
            t = (g.get(side) or {}).get("id")
            if t is not None:
                team_sched[str(t)].append((str(g.get("date"))[:10], str(g["id"])))
    for t in team_sched:
        team_sched[t].sort()
    rows = []
    for r in stats:
        g, pl, tm = r.get("game") or {}, r.get("player") or {}, r.get("team") or {}
        if g.get("id") is None or pl.get("id") is None or tm.get("id") is None:
            continue
        gid = str(g["id"])
        rows.append({"gid": gid, "date": str(g.get("date") or gmeta.get(gid, {}).get("date"))[:10], "pid": str(pl["id"]),
                     "tid": str(tm["id"]), "min": parse_minutes(r.get("min")), "row": r, "pos": pos_group(pl.get("position"))})
    teams_in_game = defaultdict(set)
    for x in rows:
        teams_in_game[x["gid"]].add(x["tid"])
    opp = {(gid, t): next(iter(ts - {t}), None) for gid, ts in teams_in_game.items() for t in ts}
    by_date = defaultdict(list)
    for x in rows:
        by_date[x["date"]].append(x)

    team_hist = defaultdict(lambda: deque(maxlen=10))
    hist = defaultdict(lambda: deque(maxlen=40))
    career = defaultdict(lambda: {"fg3a": 0.0, "fg3m": 0.0, "reb": 0.0, "chances": 0.0})
    league = {"fg3a": 0.0, "fg3m": 0.0, "reb": 0.0, "chances": 0.0}
    pos_share = defaultdict(lambda: {"d": [0.0, 0], "o": [0.0, 0]})
    last_team, roster = {}, defaultdict(set)
    played = defaultdict(set)
    out = {}
    for day in sorted(by_date):
        today = by_date[day]
        vac_cache = {}
        for x in today:
            h = list(hist[x["pid"]])
            if len(h) < 3:
                continue
            o = opp.get((x["gid"], x["tid"]))
            th, oh = list(team_hist[x["tid"]]), list(team_hist[o]) if o else []
            f = {}
            # ---- three-pointers
            g5 = [p for p in h if p["min"] >= 5]
            f["fg3a_pm"] = ewma([p["fg3a"] / p["min"] for p in g5], HL) if g5 else None
            lg = league["fg3m"] / league["fg3a"] if league["fg3a"] > 0 else 0.36
            c = career[x["pid"]]
            f["fg3_pct_shrunk"] = (c["fg3m"] + K3 * lg) / (c["fg3a"] + K3)
            a5, m5 = sum(p["fg3a"] for p in h[-5:]), sum(p["fg3m"] for p in h[-5:])
            f["hot_gap"] = (m5 / a5 - f["fg3_pct_shrunk"]) if a5 >= 10 else None
            r5 = [p["fg3a"] / p["min"] for p in g5[-5:]]
            r20 = [p["fg3a"] / p["min"] for p in g5[-20:]]
            f["attempt_trend"] = (sum(r5) / len(r5) - sum(r20) / len(r20)) if r5 and r20 else None
            # ---- rebounds: team opportunities
            if len(th) >= 5 and len(oh) >= 5:
                f["exp_dreb_opps"] = 0.5 * sum(t["own_miss"] for t in oh) / len(oh) + 0.5 * sum(t["opp_miss"] for t in th) / len(th)
                f["exp_oreb_opps"] = 0.5 * sum(t["own_miss"] for t in th) / len(th) + 0.5 * sum(t["opp_miss"] for t in oh) / len(oh)
                f["opp_oreb_rate"] = sum(t["oreb"] for t in oh) / max(1.0, sum(t["own_miss"] for t in oh))
            # player shares, shrunk toward the position mean to date
            ds = [p["dreb"] / (p["opp_miss"] * p["min"] / 48) for p in g5 if p["opp_miss"] > 0]
            os_ = [p["oreb"] / (p["own_miss"] * p["min"] / 48) for p in g5 if p["own_miss"] > 0]
            ps = pos_share[x["pos"]]
            pd_ = ps["d"][0] / ps["d"][1] if ps["d"][1] else 0.08
            po_ = ps["o"][0] / ps["o"][1] if ps["o"][1] else 0.03
            n = len(ds)
            f["dreb_share"] = ((ewma(ds, HL) * n + pd_ * KS) / (n + KS)) if ds else pd_
            n = len(os_)
            f["oreb_share"] = ((ewma(os_, HL) * n + po_ * KS) / (n + KS)) if os_ else po_
            # ---- vacated teammates (absent from the team's previous game; pregame-known)
            if x["tid"] not in vac_cache:
                sched = [s for s in team_sched.get(x["tid"], []) if s[0] < day]
                absent = []
                if sched:
                    prev_gid = sched[-1][1]
                    recent = {g_ for _, g_ in sched[-5:]}
                    for pid in roster[x["tid"]]:
                        ph = list(hist[pid])
                        if not ph or ph[-1]["gid"] not in recent:
                            continue
                        last10 = ph[-10:]
                        if sum(p["min_adj"] for p in last10) / len(last10) >= ROTATION_MIN and pid not in played.get(prev_gid, set()):
                            absent.append((pid, sum(p["reb"] for p in last10) / len(last10), sum(p["ast"] for p in last10) / len(last10)))
                vac_cache[x["tid"]] = absent
            others = [a for a in vac_cache[x["tid"]] if a[0] != x["pid"]]
            f["vacated_reb"] = sum(a[1] for a in others)
            f["vacated_ast"] = sum(a[2] for a in others)
            # ---- matchup features from R11
            mf = matchup.get((x["gid"], x["pid"]), {})
            for k in ("height_diff", "opp_fg3a_rate", "height_in", "opp_size_in"):
                f[k] = mf.get(k)
            # ---- tracking (optional)
            if tracking is not None:
                tr = [p for p in g5 if p.get("chances") is not None]
                f["chances_pm"] = ewma([p["chances"] / p["min"] for p in tr], HL) if tr else None
                lc = league["reb"] / league["chances"] if league["chances"] > 0 else 0.6
                f["chance_conv"] = (c["reb"] + KC * lc) / (c["chances"] + KC) if c["chances"] > 0 else None
                f["tracking_games"] = len(tr)
            out[(x["gid"], x["pid"])] = f
        # ---- fold today's results into history
        agg = defaultdict(lambda: {"fga": 0.0, "fgm": 0.0, "fg3a": 0.0, "oreb": 0.0, "dreb": 0.0})
        for x in today:
            a = agg[(x["gid"], x["tid"])]
            for k in a:
                a[k] += _f(x["row"], k)
        team_game = {}
        for (gid, tid), a in agg.items():
            o = opp.get((gid, tid))
            oa = agg.get((gid, o), {"fga": 0.0, "fgm": 0.0})
            team_game[(gid, tid)] = {"own_miss": a["fga"] - a["fgm"], "opp_miss": oa["fga"] - oa["fgm"],
                                     "oreb": a["oreb"], "dreb": a["dreb"], "fg3a": a["fg3a"]}
        for k, v in team_game.items():
            team_hist[k[1]].append(v)
        for x in today:
            played[x["gid"]].add(x["pid"])
        for x in today:
            r = x["row"]
            tg = team_game[(x["gid"], x["tid"])]
            tv = (tracking or {}).get((x["gid"], x["pid"]))
            ch = None
            if tv is not None and tv.get("rebound_chances_total") is not None:
                ch = float(tv["rebound_chances_total"])
            rec = {"gid": x["gid"], "min": x["min"], "min_adj": x["min"], "fg3a": _f(r, "fg3a"), "fg3m": _f(r, "fg3m"),
                   "dreb": _f(r, "dreb"), "oreb": _f(r, "oreb"), "reb": _f(r, "reb"), "ast": _f(r, "ast"),
                   "own_miss": tg["own_miss"], "opp_miss": tg["opp_miss"], "chances": ch}
            hist[x["pid"]].append(rec)
            c = career[x["pid"]]
            c["fg3a"] += rec["fg3a"]; c["fg3m"] += rec["fg3m"]
            league["fg3a"] += rec["fg3a"]; league["fg3m"] += rec["fg3m"]
            if ch is not None:
                c["reb"] += rec["reb"]; c["chances"] += ch
                league["reb"] += rec["reb"]; league["chances"] += ch
            if x["min"] >= 5:
                if tg["opp_miss"] > 0:
                    pos_share[x["pos"]]["d"][0] += rec["dreb"] / (tg["opp_miss"] * x["min"] / 48); pos_share[x["pos"]]["d"][1] += 1
                if tg["own_miss"] > 0:
                    pos_share[x["pos"]]["o"][0] += rec["oreb"] / (tg["own_miss"] * x["min"] / 48); pos_share[x["pos"]]["o"][1] += 1
            if last_team.get(x["pid"]) not in (None, x["tid"]):
                roster[last_team[x["pid"]]].discard(x["pid"])
            last_team[x["pid"]] = x["tid"]
            roster[x["tid"]].add(x["pid"])
    return out
