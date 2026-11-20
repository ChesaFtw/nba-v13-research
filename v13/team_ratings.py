"""Point-in-time team possessions, Four Factors and opponent-adjusted ratings.

Inputs are BALLDONTLIE objects already filtered to played regular-season rows
(see scripts/regular_season_filter.py):
  games: {id, date, datetime, home_team{id}, visitor_team{id}, home_team_score, visitor_team_score, status}
  stats: {min, fgm, fga, fg3m, fg3a, ftm, fta, oreb, dreb, turnover, pts, team{id}, game{id}}

Every projection for a game on date D is fitted ONLY on games dated before D.
"""
import math
from collections import defaultdict
from datetime import date

import numpy as np

BOX_FIELDS = ("pts", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "oreb", "dreb", "turnover")


def _tid(x):
    return x.get("id") if isinstance(x, dict) else x


def team_boxes(stats):
    """(game_id, team_id) -> summed box score."""
    out = defaultdict(lambda: {k: 0.0 for k in BOX_FIELDS})
    for r in stats:
        gid = (r.get("game") or {}).get("id")
        tid = _tid(r.get("team"))
        if gid is None or tid is None:
            continue
        box = out[(str(gid), str(tid))]
        for k in BOX_FIELDS:
            box[k] += float(r.get(k) or 0)
    return dict(out)


def possessions(box):
    """Standard estimate: FGA - OREB + TOV + 0.44*FTA."""
    return box["fga"] - box["oreb"] + box["turnover"] + 0.44 * box["fta"]


def four_factors(own, opp):
    poss = possessions(own)
    return {
        "efg": (own["fgm"] + 0.5 * own["fg3m"]) / own["fga"] if own["fga"] else None,
        "tov_rate": own["turnover"] / poss if poss else None,
        "orb_rate": own["oreb"] / (own["oreb"] + opp["dreb"]) if (own["oreb"] + opp["dreb"]) else None,
        "ft_rate": own["ftm"] / own["fga"] if own["fga"] else None,
        "three_rate": own["fg3a"] / own["fga"] if own["fga"] else None,
    }


def game_records(games, stats):
    """One record per finished game with both teams' box scores and possessions."""
    boxes = team_boxes(stats)
    recs = []
    for g in games:
        if g.get("status") != "Final":
            continue
        gid, h, a = str(g["id"]), str(_tid(g.get("home_team"))), str(_tid(g.get("visitor_team")))
        hb, ab = boxes.get((gid, h)), boxes.get((gid, a))
        if not hb or not ab or not hb["fga"] or not ab["fga"]:
            continue
        poss = 0.5 * (possessions(hb) + possessions(ab))
        if poss <= 0:
            continue
        recs.append({"game_id": gid, "date": str(g["date"])[:10], "season": g.get("season"),
                     "home": h, "away": a, "poss": poss,
                     "home_pts": float(g["home_team_score"]), "away_pts": float(g["visitor_team_score"]),
                     "home_ff": four_factors(hb, ab), "away_ff": four_factors(ab, hb)})
    return sorted(recs, key=lambda r: (r["date"], r["game_id"]))


class RatingModel:
    """Weighted ridge fit of offence, defence and pace with exponential time decay.

    points_per_100(team T vs opp O at home h) = mu + home*h + off[T] - def[O]
    possessions(T vs O) = pace_mu + pace[T] + pace[O]
    """

    def __init__(self, half_life_days=45.0, ridge=8.0, prior_season_weight=0.5):
        self.half_life = half_life_days
        self.ridge = ridge
        self.prior_weight = prior_season_weight

    def fit(self, records, as_of, current_season=None):
        teams = sorted({r["home"] for r in records} | {r["away"] for r in records})
        idx = {t: i for i, t in enumerate(teams)}
        n = len(teams)
        if n < 2 or not records:
            self.ok = False
            return self
        as_of_d = date.fromisoformat(as_of)
        rows_e, y_e, w_e, rows_p, y_p, w_p = [], [], [], [], [], []
        for r in records:
            age = (as_of_d - date.fromisoformat(r["date"])).days
            w = 0.5 ** (age / self.half_life)
            if current_season is not None and r.get("season") is not None and r["season"] != current_season:
                w *= self.prior_weight
            for team, opp, pts, home in ((r["home"], r["away"], r["home_pts"], 1), (r["away"], r["home"], r["away_pts"], 0)):
                x = np.zeros(2 + 2 * n)
                x[0] = 1; x[1] = home; x[2 + idx[team]] = 1; x[2 + n + idx[opp]] = -1
                rows_e.append(x); y_e.append(100 * pts / r["poss"]); w_e.append(w)
            xp = np.zeros(1 + n); xp[0] = 1; xp[1 + idx[r["home"]]] = 1; xp[1 + idx[r["away"]]] = 1
            rows_p.append(xp); y_p.append(r["poss"]); w_p.append(w)
        self.beta_e = self._solve(np.array(rows_e), np.array(y_e), np.array(w_e), unpenalised=2)
        self.beta_p = self._solve(np.array(rows_p), np.array(y_p), np.array(w_p), unpenalised=1)
        resid = np.array(y_e) - np.array(rows_e) @ self.beta_e
        self.eff_sd = float(np.sqrt(np.average(resid ** 2, weights=w_e)))
        self.idx, self.n, self.ok = idx, n, True
        return self

    def _solve(self, X, y, w, unpenalised):
        W = w[:, None]
        penalty = np.eye(X.shape[1]) * self.ridge
        penalty[:unpenalised, :unpenalised] = 0
        return np.linalg.solve(X.T @ (W * X) + penalty, X.T @ (w * y))

    def rating(self, team):
        i = self.idx.get(team)
        if i is None:
            return 0.0, 0.0, 0.0
        return float(self.beta_e[2 + i]), float(self.beta_e[2 + self.n + i]), float(self.beta_p[1 + i])

    def project(self, home, away):
        if not getattr(self, "ok", False):
            return None
        ho, hd, hp = self.rating(home)
        ao, ad, ap = self.rating(away)
        mu, hca = self.beta_e[0], self.beta_e[1]
        poss = self.beta_p[0] + hp + ap
        home_ppp = mu + hca + ho - ad
        away_ppp = mu + ao - hd
        return {"poss": float(poss), "home_pts": float(poss * home_ppp / 100), "away_pts": float(poss * away_ppp / 100),
                "home_court_per100": float(hca), "home_net": ho + hd, "away_net": ao + ad}


def rolling_four_factors(records, team, before, window=15):
    """Mean of a team's own and allowed Four Factors over its last `window` games before `before`."""
    own, allowed = defaultdict(list), defaultdict(list)
    games = [r for r in records if r["date"] < before and team in (r["home"], r["away"])][-window:]
    for r in games:
        mine, theirs = (r["home_ff"], r["away_ff"]) if r["home"] == team else (r["away_ff"], r["home_ff"])
        for k, v in mine.items():
            if v is not None: own[k].append(v)
        for k, v in theirs.items():
            if v is not None: allowed[k].append(v)
    out = {f"ff_{k}": sum(v) / len(v) for k, v in own.items() if v}
    out.update({f"ff_allowed_{k}": sum(v) / len(v) for k, v in allowed.items() if v})
    return out


def pregame_projections(games, stats, refit_every_days=1, **model_kwargs):
    """Projection per game using only games on strictly earlier dates. Multi-season input allowed."""
    recs = game_records(games, stats)
    by_date = defaultdict(list)
    for g in games:
        by_date[str(g["date"])[:10]].append(g)
    out, model, last_fit = [], None, None
    for day in sorted(by_date):
        prior = [r for r in recs if r["date"] < day]
        if len(prior) < 60:
            continue
        season = next((g.get("season") for g in by_date[day] if g.get("season") is not None), None)
        if model is None or last_fit is None or (date.fromisoformat(day) - date.fromisoformat(last_fit)).days >= refit_every_days:
            model = RatingModel(**model_kwargs).fit(prior, day, season); last_fit = day
        for g in by_date[day]:
            h, a = str(_tid(g.get("home_team"))), str(_tid(g.get("visitor_team")))
            p = model.project(h, a)
            if p is None:
                continue
            row = {"game_id": str(g["id"]), "date": day, "season": season, "home": h, "away": a,
                   "proj_poss": p["poss"], "proj_home_pts": p["home_pts"], "proj_away_pts": p["away_pts"],
                   "proj_margin": p["home_pts"] - p["away_pts"], "proj_total": p["home_pts"] + p["away_pts"],
                   "home_court_per100": p["home_court_per100"]}
            for side, team in (("home", h), ("away", a)):
                for k, v in rolling_four_factors(prior, team, day).items():
                    row[f"{side}_{k}"] = v
            if g.get("status") == "Final" and g.get("home_team_score") is not None:
                row["actual_margin"] = float(g["home_team_score"]) - float(g["visitor_team_score"])
                row["actual_total"] = float(g["home_team_score"]) + float(g["visitor_team_score"])
            out.append(row)
    return out
