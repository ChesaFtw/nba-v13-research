"""Round 13: pregame minutes / opportunity features and a minutes-distribution model.

Every feature for a player-game uses only games played on strictly earlier days. Inputs are
regular-season box scores (played rows only, min > 0) and the game index. Nothing here uses
prices, injury reports or same-game information.

Absence is never read from the current game's box score (that is only known after tip).
The pregame-safe proxy is: a rotation teammate who did NOT play in the team's PREVIOUS game.
"""
import math
from collections import defaultdict, deque
from datetime import date

import numpy as np

from scripts.regular_season_filter import minutes as parse_minutes

FEATURES = ("m_l3", "m_l5", "m_l10", "m_l20", "m_ewma", "m_sd10", "trend", "days_rest", "b2b",
            "three_in_four", "four_in_six", "missed_before", "return_flag", "vacated_min", "vacated_n",
            "top5_last", "pf_per36", "log_games")
ROTATION_MIN = 15.0      # a teammate counts as rotation if his last-10 average is at least this
EWMA_HALFLIFE = 5.0      # games
RATE_HALFLIFE = 10.0     # games, per-minute stat rates
MIN_HISTORY = 3          # player games required before features are produced
STATS = {"points": ("pts",), "rebounds": ("reb",), "assists": ("ast",), "threes": ("fg3m",), "pra": ("pts", "reb", "ast")}
QUANTILES = (0.1, 0.5, 0.9)
BUCKETS = (15.0, 25.0, 32.0)  # predicted-minutes buckets for the residual distribution


def _f(r, k):
    try:
        return float(r.get(k) or 0)
    except (TypeError, ValueError):
        return 0.0


def ot_periods(game):
    try:
        return max(0, int(game.get("period") or 4) - 4)
    except (TypeError, ValueError):
        return 0


def ewma(values, halflife):
    if not values:
        return None
    a = 1 - 0.5 ** (1 / halflife)
    s = values[0]
    for v in values[1:]:
        s = a * v + (1 - a) * s
    return s


def build(stats, games):
    """(game_id, player_id) -> {"features": {...}, "minutes": actual, "stats": {...}, "rates": {...},
    "date": ..., "season": ...}. Features and rates come only from strictly earlier days."""
    gmeta = {str(g["id"]): g for g in games}
    team_sched = defaultdict(list)  # team -> sorted [(date, gid)]
    for g in games:
        if g.get("postseason"):
            continue
        d = str(g.get("date"))[:10]
        for side in ("home_team", "visitor_team"):
            t = (g.get(side) or {}).get("id")
            if t is not None:
                team_sched[str(t)].append((d, str(g["id"])))
    for t in team_sched:
        team_sched[t].sort()
    rows = []
    for r in stats:
        g, pl, tm = r.get("game") or {}, r.get("player") or {}, r.get("team") or {}
        if g.get("id") is None or pl.get("id") is None or tm.get("id") is None:
            continue
        gid = str(g["id"])
        meta = gmeta.get(gid, {})
        rows.append({"gid": gid, "date": str(g.get("date") or meta.get("date"))[:10], "pid": str(pl["id"]),
                     "tid": str(tm["id"]), "min": parse_minutes(r.get("min")), "row": r,
                     "ot": ot_periods(meta or g), "season": meta.get("season", g.get("season"))})
    by_date = defaultdict(list)
    for x in rows:
        by_date[x["date"]].append(x)
    played = defaultdict(set)                        # gid -> set of pids who played
    game_minutes = defaultdict(dict)                 # gid -> {pid: minutes}
    hist = defaultdict(lambda: deque(maxlen=40))     # pid -> recent dicts
    last_team = {}                                   # pid -> team of most recent row
    games_seen = defaultdict(int)
    out = {}
    roster = defaultdict(set)                        # team -> pids whose most recent row is for it
    for day in sorted(by_date):
        today = by_date[day]
        vac_cache = {}
        for x in today:
            h = list(hist[x["pid"]])
            if len(h) < MIN_HISTORY:
                continue
            adj = [p["min_adj"] for p in h]
            l = lambda n: sum(adj[-n:]) / len(adj[-n:])
            sched = [s for s in team_sched.get(x["tid"], []) if s[0] < day]
            prev_gid = sched[-1][1] if sched else None
            # own missed team games since last appearance
            last_gid = h[-1]["gid"]
            missed = 0
            if h[-1]["tid"] == x["tid"]:  # after a trade, missed games for the new team are undefined: 0
                for d_, g_ in reversed(sched):
                    if g_ == last_gid:
                        break
                    missed += 1
            # rotation teammates absent from the team's previous game (pregame-known)
            if x["tid"] not in vac_cache:
                absent = []
                if prev_gid is not None:
                    recent_team_games = {g_ for _, g_ in sched[-5:]}
                    for pid in roster[x["tid"]]:
                        ph = hist[pid]
                        if not ph or ph[-1]["gid"] not in recent_team_games:
                            continue
                        avg = sum(p["min_adj"] for p in list(ph)[-10:]) / min(10, len(ph))
                        if avg >= ROTATION_MIN and pid not in played.get(prev_gid, set()):
                            absent.append((pid, avg))
                vac_cache[x["tid"]] = absent
            others = [a for pid, a in vac_cache[x["tid"]] if pid != x["pid"]]
            vac_min, vac_n = sum(others), len(others)
            top5 = 0
            if prev_gid is not None and x["pid"] in game_minutes.get(prev_gid, {}):
                ranked = sorted(game_minutes[prev_gid].items(), key=lambda kv: -kv[1])
                team_ranked = [pid for pid, _ in ranked if last_team.get(pid) == x["tid"]][:5]
                top5 = int(x["pid"] in team_ranked)
            d0 = date.fromisoformat(day)
            dates = [date.fromisoformat(d_) for d_, _ in sched]
            last_play = date.fromisoformat(h[-1]["date"])
            mins20 = [p["min"] for p in h[-20:]]
            pf20 = [p["pf"] for p in h[-20:]]
            f = {"m_l3": l(3), "m_l5": l(5), "m_l10": l(10), "m_l20": l(20), "m_ewma": ewma(adj, EWMA_HALFLIFE),
                 "m_sd10": float(np.std(adj[-10:])), "trend": l(3) - l(20),
                 "days_rest": min(5, (d0 - last_play).days),
                 "b2b": int(any((d0 - d).days == 1 for d in dates)),
                 "three_in_four": int(sum(1 for d in dates if (d0 - d).days <= 3) >= 2),
                 "four_in_six": int(sum(1 for d in dates if (d0 - d).days <= 5) >= 3),
                 "missed_before": min(10, missed), "return_flag": int(missed >= 3),
                 "vacated_min": vac_min, "vacated_n": vac_n, "top5_last": top5,
                 "pf_per36": 36 * sum(pf20) / max(1.0, sum(mins20)), "log_games": math.log1p(games_seen[x["pid"]])}
            rates = {}
            for m, keys in STATS.items():
                per = [sum(p[k] for k in keys) / p["min"] for p in h if p["min"] >= 5]
                rates[m] = ewma(per, RATE_HALFLIFE) if per else None
            out[(x["gid"], x["pid"])] = {"features": f, "minutes": x["min"], "date": day, "season": x["season"],
                                         "tid": x["tid"], "rates": rates,
                                         "stats": {m: sum(_f(x["row"], k) for k in keys) for m, keys in STATS.items()}}
        # update state with today's results only after all of today's features exist
        for x in today:
            played[x["gid"]].add(x["pid"])
            game_minutes[x["gid"]][x["pid"]] = x["min"]
        for x in today:
            adj = x["min"] * 48.0 / (48.0 + 5.0 * x["ot"])
            vals = {k: _f(x["row"], k) for k in ("pts", "reb", "ast", "fg3m")}
            hist[x["pid"]].append({"gid": x["gid"], "tid": x["tid"], "date": day, "min": x["min"], "min_adj": adj,
                                   "pf": _f(x["row"], "pf"), **vals})
            if last_team.get(x["pid"]) not in (None, x["tid"]):
                roster[last_team[x["pid"]]].discard(x["pid"])
            last_team[x["pid"]] = x["tid"]
            roster[x["tid"]].add(x["pid"])
            games_seen[x["pid"]] += 1
    return out


def design(items):
    X = np.array([[it["features"][k] for k in FEATURES] for it in items], dtype=float)
    return X


class MinutesModel:
    """Ridge regression on standardized features + empirical residual quantiles by predicted-minutes bucket."""

    def __init__(self, ridge=10.0):
        self.ridge = ridge

    def fit(self, items):
        X = design(items); y = np.array([it["minutes"] for it in items], dtype=float)
        self.mu, self.sd = X.mean(0), X.std(0)
        self.sd[self.sd < 1e-9] = 1.0
        Z = np.column_stack([np.ones(len(X)), (X - self.mu) / self.sd])
        pen = np.eye(Z.shape[1]) * self.ridge; pen[0, 0] = 0
        self.beta = np.linalg.solve(Z.T @ Z + pen, Z.T @ y)
        pred = self.predict_mean(items)
        res = y - pred
        b = np.digitize(pred, BUCKETS)
        self.res_q = {}
        for k in range(len(BUCKETS) + 1):
            r = res[b == k] if (b == k).sum() >= 50 else res
            self.res_q[k] = [float(np.quantile(r, q)) for q in QUANTILES]
        return self

    def predict_mean(self, items):
        X = design(items)
        Z = np.column_stack([np.ones(len(X)), (X - self.mu) / self.sd])
        return np.clip(Z @ self.beta, 0, 48)

    def predict_quantiles(self, items):
        pred = self.predict_mean(items)
        b = np.digitize(pred, BUCKETS)
        return np.array([[np.clip(p + self.res_q[k][i], 0, 53) for i in range(len(QUANTILES))] for p, k in zip(pred, b)])

    def coefficients(self):
        return {k: round(float(v), 4) for k, v in zip(("intercept",) + FEATURES, self.beta)}


def nb_dispersion(mu, y):
    """Method-of-moments NB dispersion phi in Var = mu + phi*mu^2 (phi >= 0)."""
    mu, y = np.asarray(mu, float), np.asarray(y, float)
    phi = float(np.mean((y - mu) ** 2 - mu) / max(1e-9, np.mean(mu ** 2)))
    return max(phi, 1e-6)


def nb_over_prob(mu, phi, line):
    """P(X > line) for X ~ NegBin(mean mu, Var = mu + phi*mu^2); line is usually a half-point."""
    mu = max(mu, 1e-6)
    r = 1.0 / phi
    p0 = (r / (r + mu)) ** r
    k_max = int(math.floor(line))
    cdf, pk = 0.0, p0
    for k in range(k_max + 1):
        cdf += pk
        pk *= (k + r) / (k + 1) * mu / (r + mu)
    return min(max(1.0 - cdf, 1e-6), 1 - 1e-6)
