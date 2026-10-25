"""Round 15 shared opportunity foundation: per-minute event rates, conversion rates, team/opponent context,
and a corrected teammate-absence signal. Strictly earlier days only (same rule as R11/R13/R14).

Per-minute EWMA rates (half-life 10 games, games with >= 5 min), box score:
  fga2_pm, fg3a_pm, fta_pm, ast_pm, stl_pm, blk_pm, tov_pm, pts_pm, reb_pm, usage_pm (FGA + 0.44 FTA + TOV)
Per-minute EWMA rates, tracking (/nba/v2/stats/advanced period 0, if supplied):
  passes_pm, touches_pm, deflections_pm, rim_contests_pm (defended_at_rim_fga)
Career-to-date conversion, shrunk to the league-to-date rate:
  fg2_pct (K=100 att), fg3_pct (K=150), ft_pct (K=50), ast_per_pass (K=200 passes),
  stl_per_defl (K=30 deflections), blk_per_contest (K=40 contests)
Team / opponent context (last 10 team games):
  pace_ratio        mean of both teams' possessions per game / league-to-date mean
  opp_def_ratio     opponent points allowed per possession / league
  opp_ast_allowed   opponent assists allowed per possession / league
  opp_tov_rate      opponent turnovers per possession / league
  opp_2pa_share     opponent (FGA - 3PA) / FGA (rim-attempt proxy)
  teammate_efg      teammates' eFG% over the team's last 10 games (player excluded)
Absence v2 (replaces R13/R14 'missed the previous game', which was wrong 39% of the time):
  For k in 1..3: rotation teammates (last-10 avg >= 15 min when last seen) who have missed the team's last
  k or more consecutive games (and at most 15): vac_usage_k (their usage volume per game), vac_ast_k.
  returning_usage / returning_ast: teammates who played the team's previous game after missing >= 3 straight.
  k is chosen on 2023-24 data only by a frozen reliability rule (scripts/r15_research.py), before any price test.
Also: pf_per36, usage_trend (usage_pm last 5 - last 20), b2b / three_in_four from v13.minutes_model.
"""
from collections import defaultdict, deque

from scripts.regular_season_filter import minutes as parse_minutes
from v13.minutes_model import ROTATION_MIN, ewma

HL = 10.0
SHRINK = {"fg2": 100.0, "fg3": 150.0, "ft": 50.0, "app": 200.0, "spd": 30.0, "bpc": 40.0}
PRIOR = {"fg2": 0.52, "fg3": 0.36, "ft": 0.78, "app": 0.10, "spd": 0.35, "bpc": 0.10}
TRACK = {"passes": "passes", "touches": "touches", "deflections": "deflections", "rim": "defended_at_rim_fga"}
KS = (1, 2, 3)


def _f(r, k):
    try:
        return float(r.get(k) or 0)
    except (TypeError, ValueError):
        return 0.0


def _tv(t, k):
    if t is None or t.get(k) is None:
        return None
    try:
        return float(t[k])
    except (TypeError, ValueError):
        return None


def build(stats, games, tracking=None):
    gmeta = {str(g["id"]): g for g in games}
    sched = defaultdict(list)
    for g in games:
        if g.get("postseason"):
            continue
        for side in ("home_team", "visitor_team"):
            t = (g.get(side) or {}).get("id")
            if t is not None:
                sched[str(t)].append((str(g.get("date"))[:10], str(g["id"])))
    for t in sched:
        sched[t].sort()
    pos_in_sched = {t: {g_: i for i, (_, g_) in enumerate(v)} for t, v in sched.items()}
    rows = []
    for r in stats:
        g, pl, tm = r.get("game") or {}, r.get("player") or {}, r.get("team") or {}
        if g.get("id") is None or pl.get("id") is None or tm.get("id") is None:
            continue
        gid = str(g["id"])
        rows.append({"gid": gid, "date": str(g.get("date") or gmeta.get(gid, {}).get("date"))[:10], "pid": str(pl["id"]),
                     "tid": str(tm["id"]), "min": parse_minutes(r.get("min")), "row": r})
    teams_in_game = defaultdict(set)
    for x in rows:
        teams_in_game[x["gid"]].add(x["tid"])
    opp = {(gid, t): next(iter(ts - {t}), None) for gid, ts in teams_in_game.items() for t in ts}
    by_date = defaultdict(list)
    for x in rows:
        by_date[x["date"]].append(x)

    hist = defaultdict(lambda: deque(maxlen=40))
    team_hist = defaultdict(lambda: deque(maxlen=10))
    career = defaultdict(lambda: defaultdict(float))
    league = defaultdict(float)
    lg_team = {"poss": 0.0, "games": 0, "pts_allowed": 0.0, "ast_allowed": 0.0, "tov": 0.0}
    last_team, roster = {}, defaultdict(set)
    last_gap = {}          # pid -> consecutive team games missed before his most recent appearance
    out = {}
    for day in sorted(by_date):
        today = by_date[day]
        ctx_cache = {}
        for x in today:
            h = list(hist[x["pid"]])
            if len(h) < 3:
                continue
            g5 = [p for p in h if p["min"] >= 5]
            f = {}
            for k in ("fga2", "fg3a", "fta", "ast", "stl", "blk", "tov", "pts", "reb", "usage"):
                f[f"{k}_pm"] = ewma([p[k] / p["min"] for p in g5], HL) if g5 else None
            for k, field in TRACK.items():
                tr = [p for p in g5 if p.get(k) is not None]
                f[f"{k}_pm" if k != "rim" else "rim_contests_pm"] = ewma([p[k] / p["min"] for p in tr], HL) if len(tr) >= 3 else None
            c = career[x["pid"]]

            def shr(num, den, key, lnum, lden):
                lg = league[lnum] / league[lden] if league[lden] > 0 else PRIOR[key]
                return (c[num] + SHRINK[key] * lg) / (c[den] + SHRINK[key])
            f["fg2_pct"] = shr("fgm2", "fga2", "fg2", "fgm2", "fga2")
            f["fg3_pct"] = shr("fg3m", "fg3a", "fg3", "fg3m", "fg3a")
            f["ft_pct"] = shr("ftm", "fta", "ft", "ftm", "fta")
            f["ast_per_pass"] = shr("ast_t", "passes", "app", "ast_t", "passes") if c["passes"] > 0 else None
            f["stl_per_defl"] = shr("stl_t", "deflections", "spd", "stl_t", "deflections") if c["deflections"] > 0 else None
            f["blk_per_contest"] = shr("blk_t", "rim", "bpc", "blk_t", "rim") if c["rim"] > 0 else None
            u5 = [p["usage"] / p["min"] for p in g5[-5:]]; u20 = [p["usage"] / p["min"] for p in g5[-20:]]
            f["usage_trend"] = (sum(u5) / len(u5) - sum(u20) / len(u20)) if u5 and u20 else None
            mins20 = [p["min"] for p in h[-20:]]
            f["pf_per36"] = 36 * sum(p["pf"] for p in h[-20:]) / max(1.0, sum(mins20))
            # team / opponent context (cached per team per day)
            if x["tid"] not in ctx_cache:
                ctx_cache[x["tid"]] = team_context(x, opp, team_hist, lg_team, sched, pos_in_sched, roster, hist, last_gap, day)
            f.update({k: v for k, v in ctx_cache[x["tid"]].items() if not k.startswith("_")})
            # absence features exclude the player himself
            for k in KS:
                mine = [a for a in ctx_cache[x["tid"]]["_absent"] if a[0] != x["pid"] and a[1] >= k]
                f[f"vac_usage_{k}"] = sum(a[2] for a in mine)
                f[f"vac_ast_{k}"] = sum(a[3] for a in mine)
            ret = [a for a in ctx_cache[x["tid"]]["_returning"] if a[0] != x["pid"]]
            f["returning_usage"] = sum(a[1] for a in ret)
            f["returning_ast"] = sum(a[2] for a in ret)
            # teammate eFG (player excluded) over team's last 10 games
            tm = [t for t in team_hist[x["tid"]]]
            if tm:
                fga = sum(t["fga"] for t in tm) - sum(p["fga2"] + p["fg3a"] for p in h if p["gid"] in {t["gid"] for t in tm})
                made = sum(t["fgm"] + 0.5 * t["fg3m"] for t in tm) - sum(p["fgm2"] + 1.5 * p["fg3m"] for p in h if p["gid"] in {t["gid"] for t in tm})
                f["teammate_efg"] = made / fga if fga > 0 else None
            out[(x["gid"], x["pid"])] = f
        # ---- fold today's results into history
        agg = defaultdict(lambda: defaultdict(float))
        for x in today:
            a = agg[(x["gid"], x["tid"])]
            for k in ("fga", "fgm", "fg3a", "fg3m", "fta", "ftm", "oreb", "tov_", "ast", "pts", "stl"):
                a[k] += _f(x["row"], "turnover" if k == "tov_" else k)
        for (gid, tid), a in agg.items():
            o = agg.get((gid, opp.get((gid, tid))), {})
            poss = a["fga"] - a["oreb"] + a["tov_"] + 0.44 * a["fta"]
            oposs = o.get("fga", 0) - o.get("oreb", 0) + o.get("tov_", 0) + 0.44 * o.get("fta", 0) if o else poss
            team_hist[tid].append({"gid": gid, "poss": poss, "pts_allowed": o.get("pts", 0.0), "ast_allowed": o.get("ast", 0.0),
                                   "opp_poss": oposs, "tov": a["tov_"], "fga": a["fga"], "fgm": a["fgm"], "fg3a": a["fg3a"],
                                   "fg3m": a["fg3m"]})
            lg_team["poss"] += poss; lg_team["games"] += 1
            lg_team["pts_allowed"] += o.get("pts", 0.0); lg_team["ast_allowed"] += o.get("ast", 0.0); lg_team["tov"] += a["tov_"]
        for x in today:
            r = x["row"]
            tv = (tracking or {}).get((x["gid"], x["pid"]))
            fga, fg3a, fgm, fg3m = _f(r, "fga"), _f(r, "fg3a"), _f(r, "fgm"), _f(r, "fg3m")
            rec = {"gid": x["gid"], "min": x["min"], "min_adj": x["min"], "fga2": fga - fg3a, "fg3a": fg3a, "fgm2": fgm - fg3m,
                   "fg3m": fg3m, "fta": _f(r, "fta"), "ftm": _f(r, "ftm"), "ast": _f(r, "ast"), "stl": _f(r, "stl"),
                   "blk": _f(r, "blk"), "tov": _f(r, "turnover"), "pts": _f(r, "pts"), "reb": _f(r, "reb"), "pf": _f(r, "pf")}
            rec["usage"] = fga + 0.44 * rec["fta"] + rec["tov"]
            for k, field in TRACK.items():
                rec[k] = _tv(tv, field)
            hist[x["pid"]].append(rec)
            c = career[x["pid"]]
            for k in ("fga2", "fgm2", "fg3a", "fg3m", "fta", "ftm"):
                c[k] += rec[k]; league[k] += rec[k]
            if rec["passes"] is not None:
                c["passes"] += rec["passes"]; c["ast_t"] += rec["ast"]; league["passes"] += rec["passes"]; league["ast_t"] += rec["ast"]
            if rec["deflections"] is not None:
                c["deflections"] += rec["deflections"]; c["stl_t"] += rec["stl"]
                league["deflections"] += rec["deflections"]; league["stl_t"] += rec["stl"]
            if rec["rim"] is not None:
                c["rim"] += rec["rim"]; c["blk_t"] += rec["blk"]; league["rim"] += rec["rim"]; league["blk_t"] += rec["blk"]
            # gap before this appearance (consecutive team games missed)
            prev = hist[x["pid"]][-2]["gid"] if len(hist[x["pid"]]) >= 2 else None
            idx = pos_in_sched.get(x["tid"], {})
            if prev is not None and prev in idx and x["gid"] in idx:
                last_gap[x["pid"]] = max(0, idx[x["gid"]] - idx[prev] - 1)
            else:
                last_gap[x["pid"]] = 0
            if last_team.get(x["pid"]) not in (None, x["tid"]):
                roster[last_team[x["pid"]]].discard(x["pid"])
            last_team[x["pid"]] = x["tid"]
            roster[x["tid"]].add(x["pid"])
    return out


def team_context(x, opp, team_hist, lg, sched, pos_in_sched, roster, hist, last_gap, day):
    o = opp.get((x["gid"], x["tid"]))
    th, oh = list(team_hist[x["tid"]]), list(team_hist[o]) if o else []
    ctx = {}
    lg_poss = lg["poss"] / lg["games"] if lg["games"] else None
    if len(th) >= 5 and len(oh) >= 5 and lg_poss:
        own = sum(t["poss"] for t in th) / len(th); op = sum(t["poss"] for t in oh) / len(oh)
        ctx["pace_ratio"] = ((own + op) / 2) / lg_poss
        lg_def = lg["pts_allowed"] / lg["poss"]; lg_ast = lg["ast_allowed"] / lg["poss"]; lg_tov = lg["tov"] / lg["poss"]
        opp_poss_faced = sum(t["opp_poss"] for t in oh)
        ctx["opp_def_ratio"] = (sum(t["pts_allowed"] for t in oh) / max(1.0, opp_poss_faced)) / lg_def if lg_def else None
        ctx["opp_ast_allowed"] = (sum(t["ast_allowed"] for t in oh) / max(1.0, opp_poss_faced)) / lg_ast if lg_ast else None
        ctx["opp_tov_rate"] = (sum(t["tov"] for t in oh) / max(1.0, sum(t["poss"] for t in oh))) / lg_tov if lg_tov else None
        ctx["opp_2pa_share"] = sum(t["fga"] - t["fg3a"] for t in oh) / max(1.0, sum(t["fga"] for t in oh))
    # absence v2 and returning teammates (pregame-known from schedule + appearances)
    s = [g_ for d_, g_ in sched.get(x["tid"], []) if d_ < day]
    idx = pos_in_sched.get(x["tid"], {})
    absent, returning = [], []
    if s:
        n_before = len(s)
        prev_gid = s[-1]
        for pid in roster[x["tid"]]:
            ph = list(hist[pid])
            if not ph:
                continue
            last_gid = ph[-1]["gid"]
            if last_gid not in idx:
                continue
            missed = n_before - 1 - idx[last_gid]
            last10 = ph[-10:]
            avg_min = sum(p["min"] for p in last10) / len(last10)
            use = sum(p["usage"] for p in last10) / len(last10); ast = sum(p["ast"] for p in last10) / len(last10)
            if avg_min < ROTATION_MIN:
                continue
            if 1 <= missed <= 15:
                absent.append((pid, missed, use, ast))
            elif missed == 0 and last_gid == prev_gid and last_gap.get(pid, 0) >= 3:
                pre = ph[-11:-1] or ph[-1:]
                returning.append((pid, sum(p["usage"] for p in pre) / len(pre), sum(p["ast"] for p in pre) / len(pre)))
    ctx["_absent"], ctx["_returning"] = absent, returning
    return ctx
