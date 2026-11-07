"""Export out-of-sample player-prop candidates (both sides) in the universal ranking schema.

Walk-forward: for each calendar month, the per-market Ridge projection is fitted only on
rows dated before that month (across all supplied seasons). Lines must be pre-tip and
carry real two-sided prices (v2 rules). Probabilities: Poisson for count markets,
normal (training-residual sigma) for points/PRA. This is the BASELINE probability model;
better models (minutes, negative binomial) plug in by replacing `probability()`.

python -m scripts.export_prop_candidates \
  --season 2025 --features props_2025.jsonl --lines opening_props_2025.jsonl --games games_2025.jsonl \
  [--season 2024 --features ... --lines ... --games ...] --out research_results/candidates/props.jsonl
"""
import argparse, json, math
from collections import Counter
from pathlib import Path

from scripts.backtest_prop_holdout_v2 import COUNT_MARKETS, dataset, fit, read
from v13.pricing import tip_times


def poisson_pmf(k, mu):
    return math.exp(-mu + k * math.log(mu) - math.lgamma(k + 1)) if k >= 0 else 0.0


def probability(proj, line, market, sigma):
    """(P(over), P(push)) for one line."""
    mu = max(proj, 1e-6)
    if market in COUNT_MARKETS:
        k = math.floor(line)
        cdf = sum(poisson_pmf(i, mu) for i in range(k + 1))
        push = poisson_pmf(int(line), mu) if float(line).is_integer() else 0.0
        over = 1 - cdf
        return over, push
    z_hi = (line + 0.5 - mu) / sigma if float(line).is_integer() else (line - mu) / sigma
    over = 0.5 * math.erfc(z_hi / math.sqrt(2))
    push = 0.0
    if float(line).is_integer():
        z_lo = (line - 0.5 - mu) / sigma
        push = 0.5 * math.erfc(z_lo / math.sqrt(2)) - over
    return over, push


def export(seasons, validated=False):
    from v13.provenance import evidence_class, require_eligible
    rows, coverage = [], Counter()
    for s in seasons:
        if validated:
            require_eligible(s["lines"], f"prop candidates season {s['season']}")
        cov = Counter()
        for r in dataset(s["features"], s["lines"], tip_times(s["games"]), cov):
            if r["over_odds"] is None or r["under_odds"] is None:
                cov["missing_price"] += 1
                continue
            rows.append({**r, "season": s["season"]})
        coverage.update({f"{s['season']}_{k}": v for k, v in cov.items()})
    rows.sort(key=lambda r: r["date"])
    out = []
    for market in sorted({r["market"] for r in rows}):
        sub = [r for r in rows if r["market"] == market]
        for month in sorted({r["date"][:7] for r in sub}):
            train = [r for r in sub if r["date"][:7] < month]
            test = [r for r in sub if r["date"][:7] == month]
            if len(train) < 500:
                coverage[f"{market}_{month}_skipped_burn_in"] += len(test)
                continue
            predictor, sigma = fit(train)
            for r, proj in zip(test, predictor(test)):
                over, push = probability(proj, r["line"], market, sigma)
                no_push = max(1 - push, 1e-9)
                delta = r["actual"] - r["line"]
                for side, price, opp, p_side in (("over", r["over_odds"], r["under_odds"], over),
                                                 ("under", r["under_odds"], r["over_odds"], 1 - over - push)):
                    outcome = "push" if abs(delta) < 1e-9 else ("win" if (delta > 0) == (side == "over") else "loss")
                    out.append({"candidate_id": f"{r['game_id']}:{r['player_id']}:{market}:{side}",
                                "season": r["season"], "date": r["date"], "game_id": r["game_id"],
                                "player_id": r["player_id"], "market": f"prop_{market}", "side": side,
                                "line": r["line"], "price": price, "opp_price": opp,
                                "p_model": min(max(p_side / no_push, 1e-4), 1 - 1e-4), "p_push": push,
                                "projection": round(proj, 3), "model_version": "ridge_rolling_poisson_v1",
                                "vendor": r["vendor"], "opened_at": r["opened_at"], "outcome": outcome,
                                "evidence_class": evidence_class(validated)})
    return out, dict(coverage)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--season", type=int, action="append", required=True)
    p.add_argument("--features", action="append", required=True)
    p.add_argument("--lines", action="append", required=True)
    p.add_argument("--games", action="append", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--validated", action="store_true", help="require provenance-gate stamped lines (fail closed)")
    a = p.parse_args()
    if not len(a.season) == len(a.features) == len(a.lines) == len(a.games):
        raise SystemExit("Give --season/--features/--lines/--games once per season, in the same order")
    seasons = [{"season": s, "features": read(f), "lines": read(l), "games": read(g)}
               for s, f, l, g in zip(a.season, a.features, a.lines, a.games)]
    cands, coverage = export(seasons, validated=a.validated)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(c) + "\n" for c in cands))
    print(json.dumps({"candidates": len(cands), "by_market": Counter(c["market"] for c in cands), "coverage": coverage}, indent=2))


if __name__ == "__main__":
    main()
