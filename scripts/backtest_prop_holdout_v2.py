"""Prop holdout backtest v2: real prices, pre-tip guard, nested selection, cross-season mode.

Differences from backtest_prop_holdout.py (kept unchanged for reproducibility):
  1. Grades at the actual opening over/under price in the source record (--pricing actual).
     Rows with no usable price are excluded and counted, never filled with -110.
  2. Drops any line whose opened_at is not strictly before tip (needs --games). Fail closed.
  3. Threshold is chosen on an inner validation slice of TRAIN using out-of-sample
     projections, not in-sample fitted values.
  4. Stakes are 1% of the bankroll at the START of each betting day (no intraday compounding
     on results that would not yet be known).
  5. Two selectors are reported side by side: raw point edge (legacy) and expected value
     from a count/normal distribution priced against the actual offered odds.
  6. Optional cross-season mode: train on a complete prior season, test on the full target
     season. This is the multi-season validation the within-season split cannot provide.

Example (within-season):
  python -m scripts.backtest_prop_holdout_v2 --features props_2025.jsonl --lines opening_props_2025.jsonl \
      --games games_2025.jsonl --out research_results/prop_backtest_v2/holdout_2025.json
Cross-season (train 2024-25, test 2025-26):
  ... --prior-features props_2024.jsonl --prior-lines opening_props_2024.jsonl --prior-games games_2024.jsonl
"""
import argparse, json, math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge

from scripts.backtest_prop_holdout import ALIASES, FEATURES, MARKETS, VENDORS
from v13.pricing import (ASSUMED_110, implied_probability, no_vig_pair, pregame_status,
                         profit_per_unit, prop_prices, tip_times, wilson)

COUNT_MARKETS = {"threes", "assists", "rebounds"}
EDGE_GRID = {"points": [2, 3, 4, 5], "rebounds": [1, 1.5, 2, 2.5], "assists": [1, 1.5, 2],
             "threes": [.5, 1, 1.5], "pra": [3, 4, 5, 6]}
EV_GRID = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10]
MIN_SELECTION_BETS = 40


def read(path):
    return [json.loads(x) for x in Path(path).open() if x.strip()]


def normalize_lines(rows, tips, coverage):
    """One record per (game, player, market): vendor priority, then earliest PREGAME row."""
    selected = {}
    for r in rows:
        if (r.get("market") or {}).get("type") != "over_under":
            continue
        market = ALIASES.get(r.get("prop_type"))
        if market not in MARKETS or r.get("line_value") is None:
            continue
        try:
            line = float(r["line_value"])
        except (TypeError, ValueError):
            continue
        if not math.isfinite(line):
            continue
        status = pregame_status(r.get("opened_at"), tips.get(str(r.get("game_id"))))
        if status != "pregame":
            coverage[f"line_{status}"] += 1
            continue
        key = (str(r["game_id"]), str(r["player_id"]), market)
        vendor = str(r.get("vendor") or "").lower()
        rank = (VENDORS.index(vendor) if vendor in VENDORS else 99, str(r["opened_at"]), str(r.get("id", "")))
        if key in selected:
            coverage["duplicate_pregame_rows"] += 1
            if abs(selected[key][2] - line) > 1e-9:
                coverage["conflicting_line_values"] += 1
        if key not in selected or rank < selected[key][0]:
            selected[key] = (rank, r, line)
    return {k: (v[1], v[2]) for k, v in selected.items()}


def dataset(features, lines, tips, coverage):
    chosen = normalize_lines(lines, tips, coverage)
    out = []
    for f in features:
        for market in MARKETS:
            found = chosen.get((str(f["game_id"]), str(f["player_id"]), market))
            if found is None:
                continue
            raw, line = found
            try:
                x = [float(f[k]) for k in FEATURES(market)]
                actual = float(f[f"{market}_actual"])
            except (KeyError, TypeError, ValueError):
                continue
            over, under = prop_prices(raw)
            out.append({"date": f["game_date"], "game_id": str(f["game_id"]), "player_id": str(f["player_id"]),
                        "market": market, "line": line, "actual": actual, "x": x,
                        "over_odds": over, "under_odds": under,
                        "vendor": raw.get("vendor"), "opened_at": raw.get("opened_at")})
    return out


def p_over(projection, line, market, sigma):
    """P(actual > line). Poisson for count markets, normal otherwise. Half-lines only."""
    mu = max(projection, 1e-6)
    if market in COUNT_MARKETS:
        k = math.floor(line)  # P(X > line) = 1 - P(X <= floor(line))
        term = math.exp(-mu); cdf = term
        for i in range(1, k + 1):
            term *= mu / i; cdf += term
        return min(max(1 - cdf, 1e-6), 1 - 1e-6)
    z = (line - mu) / max(sigma, 1e-6)
    return min(max(0.5 * math.erfc(z / math.sqrt(2)), 1e-6), 1 - 1e-6)


def price_for(r, side, pricing):
    if pricing == "assumed110":
        return ASSUMED_110
    return r["over_odds"] if side == "over" else r["under_odds"]


def decide(r, selector, threshold, pricing, sigma):
    """Return (side, score) if the row qualifies, else None."""
    if r["line"] != math.floor(r["line"]) + 0.5 and selector == "ev":
        return None  # integer lines need push-aware pricing; excluded from the EV selector
    if selector == "edge":
        diff = r["projection"] - r["line"]
        if abs(diff) < threshold:
            return None
        return ("over" if diff > 0 else "under"), abs(diff)
    po = p_over(r["projection"], r["line"], r["market"], sigma)
    best = None
    for side, p in (("over", po), ("under", 1 - po)):
        price = price_for(r, side, pricing)
        if price is None:
            continue
        ev = p * profit_per_unit(price) - (1 - p)
        if ev >= threshold and (best is None or ev > best[1]):
            best = (side, ev)
    return best


def simulate(rows, selector, threshold, pricing, sigma, bankroll=10000.0):
    ledger, peak, skipped = [], bankroll, Counter()
    bydate = defaultdict(list)
    for r in rows:
        bydate[r["date"]].append(r)
    for day in sorted(bydate):
        day_start = bankroll  # stakes sized before any of today's results are known
        for r in sorted(bydate[day], key=lambda r: (r["game_id"], r["player_id"])):
            pick = decide(r, selector, threshold, pricing, sigma)
            if pick is None:
                continue
            side, score = pick
            price = price_for(r, side, pricing)
            if price is None:
                skipped["missing_price"] += 1
                continue
            delta = r["actual"] - r["line"]
            outcome = "push" if abs(delta) < 1e-9 else ("win" if (delta > 0) == (side == "over") else "loss")
            stake = round(day_start * 0.01, 2)
            pnl = round(stake * (profit_per_unit(price) if outcome == "win" else -1 if outcome == "loss" else 0), 2)
            bankroll = round(bankroll + pnl, 2); peak = max(peak, bankroll)
            market_prob = None
            if r["over_odds"] is not None and r["under_odds"] is not None:
                po_m, pu_m = no_vig_pair(r["over_odds"], r["under_odds"])
                market_prob = po_m if side == "over" else pu_m
            ledger.append({"date": r["date"], "game_id": r["game_id"], "player_id": r["player_id"],
                           "market": r["market"], "vendor": r["vendor"], "opened_at": r["opened_at"],
                           "line": r["line"], "projection": round(r["projection"], 3), "side": side,
                           "price": price, "over_odds": r["over_odds"], "under_odds": r["under_odds"],
                           "score": round(score, 4), "market_prob": market_prob,
                           "outcome": outcome, "stake": stake, "pnl": pnl, "bankroll": bankroll,
                           "drawdown_pct": round(100 * (peak - bankroll) / peak, 3)})
    return summarize(ledger, skipped)


def summarize(ledger, skipped=None):
    c = Counter(x["outcome"] for x in ledger); n = c["win"] + c["loss"]
    fair = [x for x in ledger if x["outcome"] in ("win", "loss") and x["market_prob"] is not None]
    excess = None
    if fair:
        exp = sum(x["market_prob"] for x in fair); var = sum(x["market_prob"] * (1 - x["market_prob"]) for x in fair)
        wins = sum(x["outcome"] == "win" for x in fair)
        excess = {"bets": len(fair), "market_expected_hit_rate": round(exp / len(fair), 4),
                  "realized_hit_rate": round(wins / len(fair), 4),
                  "excess": round((wins - exp) / len(fair), 4),
                  "z": round((wins - exp) / math.sqrt(var), 2) if var > 0 else None}
    staked = sum(x["stake"] for x in ledger)
    sides = {s: dict(Counter(x["outcome"] for x in ledger if x["side"] == s)) for s in ("over", "under")}
    return {"bets": len(ledger), "wins": c["win"], "losses": c["loss"], "pushes": c["push"],
            "hit_rate": round(c["win"] / n, 4) if n else None, "hit_rate_95ci": wilson(c["win"], n),
            "mean_breakeven": round(sum(implied_probability(x["price"]) for x in ledger) / len(ledger), 4) if ledger else None,
            "roi_on_turnover_pct": round(100 * sum(x["pnl"] for x in ledger) / staked, 2) if staked else None,
            "ending_bankroll": ledger[-1]["bankroll"] if ledger else 10000.0,
            "max_drawdown_pct": max((x["drawdown_pct"] for x in ledger), default=0),
            "by_side": sides, "vs_devigged_market": excess,
            "skipped": dict(skipped or {}), "ledger": ledger}


def fit(train):
    X = np.array([r["x"] for r in train]); y = np.array([r["actual"] for r in train])
    mu = X.mean(axis=0); sd = X.std(axis=0); sd[sd < 1e-8] = 1
    model = Ridge(alpha=25.0).fit((X - mu) / sd, y)
    resid = y - model.predict((X - mu) / sd)
    return (lambda rows: [float(v) for v in model.predict((np.array([r["x"] for r in rows]) - mu) / sd)]), float(resid.std())


def project(rows, predictor):
    if rows:
        for r, v in zip(rows, predictor(rows)):
            r["projection"] = v


def evaluate_market(train, test, market, pricing):
    if len(train) < 150 or len(test) < 50:
        return {"status": "insufficient coverage", "train": len(train), "test": len(test)}
    dates = sorted({r["date"] for r in train})
    inner_cut = dates[int(len(dates) * 0.7)]
    inner = [dict(r) for r in train if r["date"] < inner_cut]
    valid = [dict(r) for r in train if r["date"] >= inner_cut]
    if len(inner) < 100 or len(valid) < 50 or len(test) < 50:
        return {"status": "insufficient coverage", "train": len(train), "test": len(test)}
    predictor, sigma = fit(inner)
    project(valid, predictor)  # out-of-sample projections for threshold choice
    out = {"status": "evaluated", "train_rows": len(train), "validation_rows": len(valid), "test_rows": len(test)}
    final_predictor, final_sigma = fit(train)
    test = [dict(r) for r in test]; project(test, final_predictor)
    for selector, grid in (("edge", EDGE_GRID[market]), ("ev", EV_GRID)):
        scored = [(t, simulate(valid, selector, t, pricing, sigma)) for t in grid]
        eligible = [(t, s) for t, s in scored if s["bets"] >= MIN_SELECTION_BETS]
        if not eligible:
            out[selector] = {"status": "no qualifying validation threshold"}; continue
        t, val = max(eligible, key=lambda item: (item[1]["roi_on_turnover_pct"], -item[1]["bets"]))
        held = simulate(test, selector, t, pricing, final_sigma)
        out[selector] = {"threshold": t,
                         "validation": {k: v for k, v in val.items() if k != "ledger"},
                         "holdout": {k: v for k, v in held.items() if k != "ledger"},
                         "holdout_ledger": held["ledger"]}
    return out


def evaluate(features, lines, games, pricing="actual", prior=None):
    if not tip_times(games):
        raise RuntimeError("No tip times in games index; cannot verify pregame lines")
    coverage = Counter()
    rows = dataset(features, lines, tip_times(games), coverage)
    report = {"pricing": pricing, "line_rule": "vendor priority, earliest row with opened_at < tip",
              "coverage": dict(coverage), "matched_rows": len(rows), "markets": {}}
    if prior:
        prior_cov = Counter()
        train_all = dataset(prior["features"], prior["lines"], tip_times(prior["games"]), prior_cov)
        report.update(mode="cross_season", prior_coverage=dict(prior_cov), prior_rows=len(train_all))
        test_all = rows
    else:
        dates = sorted({r["date"] for r in rows})
        if len(dates) < 30:
            raise RuntimeError("Insufficient dated rows")
        cutoff = dates[int(len(dates) * 0.65)]
        report.update(mode="within_season", holdout_start=cutoff)
        train_all = [r for r in rows if r["date"] < cutoff]
        test_all = [r for r in rows if r["date"] >= cutoff]
    priced = sum(r["over_odds"] is not None and r["under_odds"] is not None for r in rows)
    report["rows_with_both_prices"] = priced
    for market in MARKETS:
        report["markets"][market] = evaluate_market(
            [r for r in train_all if r["market"] == market], [r for r in test_all if r["market"] == market],
            market, pricing)
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", required=True); p.add_argument("--lines", required=True)
    p.add_argument("--games", required=True); p.add_argument("--out", required=True)
    p.add_argument("--pricing", choices=("actual", "assumed110"), default="actual")
    p.add_argument("--prior-features"); p.add_argument("--prior-lines"); p.add_argument("--prior-games")
    a = p.parse_args()
    prior = None
    if a.prior_features or a.prior_lines or a.prior_games:
        if not (a.prior_features and a.prior_lines and a.prior_games):
            raise SystemExit("Cross-season mode needs --prior-features, --prior-lines and --prior-games")
        prior = {"features": read(a.prior_features), "lines": read(a.prior_lines), "games": read(a.prior_games)}
    result = evaluate(read(a.features), read(a.lines), read(a.games), a.pricing, prior)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    brief = {k: v for k, v in result.items() if k != "markets"}
    brief["markets"] = {m: {s: (v.get(s) or {}).get("holdout", (v.get(s) or {}).get("status"))
                            for s in ("edge", "ev")} if v.get("status") == "evaluated" else v
                        for m, v in result["markets"].items()}
    for m in brief["markets"].values():
        for s in ("edge", "ev"):
            if isinstance(m.get(s), dict): m[s] = {k: x for k, x in m[s].items() if k != "ledger"}
    print(json.dumps(brief, indent=2))


if __name__ == "__main__":
    main()
