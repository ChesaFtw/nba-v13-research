"""Spread and total models v2: possession-based ratings + market-residual, walk-forward.

Three model families are exported as universal-schema candidates so the ranking engine
and the reports compare them on identical games:

  direct   : P from the ratings projection alone (margin ~ N(proj_margin, sd)).
  residual : Ridge on (actual - market) using (projection - market), rest, travel,
             altitude and Four Factors matchup features; P from N(market + r_hat, sd).
  market   : the de-vigged opening price itself (zero-edge benchmark; never selected).

Every fit uses only games dated before the month being scored. Lines must be opened
before tip and carry real prices; anything else is excluded and counted.

python -m scripts.game_model_v2 \
  --games games_2023.jsonl --games games_2024.jsonl --games games_2025.jsonl \
  --stats regular_players_2023.jsonl --stats regular_players_2024.jsonl --stats regular_players_2025.jsonl \
  --odds opening_2023.jsonl --odds opening_2024.jsonl --odds opening_2025.jsonl \
  --out-candidates research_results/candidates/games.jsonl --out-report research_results/game_model_v2/report.json
"""
import argparse, json, math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from scripts.backtest_opening_spreads_totals import choose
from v13.pricing import game_price_pair, no_vig_pair, tip_times
from v13.schedule_travel import schedule_travel_features
from v13.team_ratings import pregame_projections

RIDGE_ALPHA = 50.0
MIN_TRAIN = 300


def read(path):
    return [json.loads(x) for x in Path(path).open() if x.strip()]


def _f(row, key, default=0.0):
    v = row.get(key)
    return default if v is None else float(v)


FEATURE_NAMES = {
    "spread": ["proj_minus_market", "rest_diff", "home_b2b", "away_b2b", "away_travel_1000km", "away_abs_tz_shift",
               "away_at_altitude", "efg_matchup_diff", "tov_rate_diff", "orb_rate_diff"],
    "total": ["proj_minus_market", "b2b_count", "away_at_altitude", "three_rate_sum", "ft_rate_sum", "orb_rate_sum"],
}


def residual_features(market, r):
    if market == "spread":
        return [r["proj_margin"] - r["market_value"],
                _f(r, "rest_diff"), _f(r, "home_b2b"), _f(r, "away_b2b"),
                _f(r, "away_travel_km") / 1000, abs(_f(r, "away_tz_shift")), _f(r, "away_at_altitude"),
                _f(r, "home_ff_efg") - _f(r, "away_ff_allowed_efg") - (_f(r, "away_ff_efg") - _f(r, "home_ff_allowed_efg")),
                _f(r, "home_ff_tov_rate") - _f(r, "away_ff_tov_rate"),
                _f(r, "home_ff_orb_rate") - _f(r, "away_ff_orb_rate")]
    return [r["proj_total"] - r["market_value"],
            _f(r, "home_b2b") + _f(r, "away_b2b"), _f(r, "away_at_altitude"),
            _f(r, "home_ff_three_rate") + _f(r, "away_ff_three_rate"),
            _f(r, "home_ff_ft_rate") + _f(r, "away_ff_ft_rate"),
            _f(r, "home_ff_orb_rate") + _f(r, "away_ff_orb_rate")]


def build_rows(games, stats, odds):
    proj = {p["game_id"]: p for p in pregame_projections(games, stats)}
    sched = schedule_travel_features(games)
    lines = choose(odds, tip_times(games))
    coverage, rows = Counter(), []
    for gid, p in proj.items():
        o = lines.get(gid)
        if o is None:
            coverage["no_pregame_line"] += 1; continue
        if "actual_margin" not in p:
            coverage["no_result"] += 1; continue
        for market in ("spread", "total"):
            raw = o.get("spread_home_value" if market == "spread" else "total_value")
            prices = game_price_pair(o, market)
            if raw is None or None in prices:
                coverage[f"{market}_missing_line_or_price"] += 1; continue
            line = float(raw)
            market_value = -line if market == "spread" else line  # implied home margin / total
            rows.append({**p, **sched.get(gid, {}), "market": market, "line": line, "market_value": market_value,
                         "price_a": prices[0], "price_b": prices[1], "vendor": o.get("vendor"),
                         "opened_at": o.get("opened_at"),
                         "actual": p["actual_margin"] if market == "spread" else p["actual_total"]})
    return sorted(rows, key=lambda r: (r["date"], r["game_id"], r["market"])), coverage


def normal_sides(mean, threshold, sd, integer_line):
    """P(X > threshold), P(push) for X ~ N(mean, sd) graded on a (possibly integer) line."""
    def above(t): return 0.5 * math.erfc((t - mean) / (sd * math.sqrt(2)))
    if integer_line:
        p_over, p_push = above(threshold + 0.5), above(threshold - 0.5) - above(threshold + 0.5)
    else:
        p_over, p_push = above(threshold), 0.0
    return p_over, p_push


def candidates_for(r, mean, sd, version):
    integer = float(r["line"]).is_integer()
    p_a, push = normal_sides(mean, r["market_value"], sd, integer)
    no_push = max(1 - push, 1e-9)
    sides = ("home", "away") if r["market"] == "spread" else ("over", "under")
    diff = r["actual"] - r["market_value"]
    out = []
    for side, price, opp, p in ((sides[0], r["price_a"], r["price_b"], p_a), (sides[1], r["price_b"], r["price_a"], 1 - p_a - push)):
        outcome = "push" if abs(diff) < 1e-9 else ("win" if (diff > 0) == (side == sides[0]) else "loss")
        out.append({"candidate_id": f"{r['game_id']}:{r['market']}:{side}:{version}", "season": r.get("season"),
                    "date": r["date"], "game_id": r["game_id"], "player_id": None, "market": r["market"],
                    "model_version": version, "side": side, "line": r["line"], "price": price, "opp_price": opp,
                    "p_model": min(max(p / no_push, 1e-4), 1 - 1e-4), "p_push": push, "outcome": outcome,
                    "vendor": r["vendor"], "opened_at": r["opened_at"]})
    return out


def ridge_fit(X, y, alpha=RIDGE_ALPHA):
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd < 1e-9] = 1
    Z = (X - mu) / sd
    Z1 = np.hstack([np.ones((len(Z), 1)), Z])
    pen = np.eye(Z1.shape[1]) * alpha; pen[0, 0] = 0
    beta = np.linalg.solve(Z1.T @ Z1 + pen, Z1.T @ y)
    return lambda Xn: np.hstack([np.ones((len(Xn), 1)), (Xn - mu) / sd]) @ beta, beta


def run(rows):
    cands, report = [], {"markets": {}}
    for market in ("spread", "total"):
        sub = [r for r in rows if r["market"] == market]
        err = defaultdict(list)
        coefs = {}
        for month in sorted({r["date"][:7] for r in sub}):
            train = [r for r in sub if r["date"][:7] < month]
            test = [r for r in sub if r["date"][:7] == month]
            if len(train) < MIN_TRAIN:
                continue
            key = "proj_margin" if market == "spread" else "proj_total"
            direct_sd = float(np.std([r["actual"] - r[key] for r in train]))
            X = np.array([residual_features(market, r) for r in train])
            y = np.array([r["actual"] - r["market_value"] for r in train])
            predict, beta = ridge_fit(X, y)
            resid_sd = float(np.std(y - predict(X)))
            coefs[month] = [round(float(b), 4) for b in beta]
            r_hat = predict(np.array([residual_features(market, r) for r in test]))
            for r, rh in zip(test, r_hat):
                mean_resid = r["market_value"] + float(rh)
                cands += candidates_for(r, r[key], direct_sd, "direct_v2")
                cands += candidates_for(r, mean_resid, resid_sd, "residual_v2")
                season = str(r.get("season"))
                err[(season, "market")].append((r["actual"] - r["market_value"]) ** 2)
                err[(season, "direct")].append((r["actual"] - r[key]) ** 2)
                err[(season, "residual")].append((r["actual"] - mean_resid) ** 2)
        report["markets"][market] = {
            "out_of_sample_mse": {f"{s}_{m}": round(float(np.mean(v)), 3) for (s, m), v in sorted(err.items())},
            "out_of_sample_games": {s: len(v) for (s, m), v in sorted(err.items()) if m == "market"},
            "residual_coefficients_last_fit": coefs[max(coefs)] if coefs else None,
            "feature_order": ["intercept"] + FEATURE_NAMES[market],
            "reading_guide": "A model adds information only if its MSE is below the market's in every season."}
    return cands, report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--games", action="append", required=True)
    p.add_argument("--stats", action="append", required=True)
    p.add_argument("--odds", action="append", required=True)
    p.add_argument("--out-candidates", required=True); p.add_argument("--out-report", required=True)
    a = p.parse_args()
    games = [g for path in a.games for g in read(path) if g.get("postseason") is False]
    stats = [s for path in a.stats for s in read(path)]
    odds = [o for path in a.odds for o in read(path)]
    rows, coverage = build_rows(games, stats, odds)
    cands, report = run(rows)
    report["coverage"] = dict(coverage); report["rows"] = len(rows)
    for path, content in ((a.out_candidates, "".join(json.dumps(c) + "\n" for c in cands)),
                          (a.out_report, json.dumps(report, indent=2) + "\n")):
        Path(path).parent.mkdir(parents=True, exist_ok=True); Path(path).write_text(content)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
