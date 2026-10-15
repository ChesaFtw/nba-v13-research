"""Cross-market selection, bet-count sweep and staking comparison on universal candidates.

1. Walk-forward scoring: each month's candidates are scored with per-market calibrators
   (model + de-vigged market blend) fitted only on earlier months.
2. For each target count N, the EV cutoff is chosen on the VALIDATION season's scored
   candidates, then frozen and applied to the TEST season.
3. Flat 1% and quarter-Kelly tiers (1/2/3%) are simulated on the same frozen selections,
   with day-start staking and a day-bootstrap Monte Carlo.

python -m scripts.rank_and_stake --candidates props.jsonl --candidates games.jsonl \
    --models residual_v2,ridge_rolling_poisson_v1 --validation-season 2024 --test-season 2025 \
    --out research_results/ranking/report.json
"""
import argparse, json
from collections import Counter
from pathlib import Path

from v13.ranking import (STAKING, monte_carlo, parametric_monte_carlo, reliability, select, simulate,
                         tau_for_target, walk_forward_score)

SHRINKS = (1.0, 0.5, 0.25, 0.0)


def selection_calibration(chosen):
    """Winner's-curse check: do the SELECTED bets win as often as the engine predicted?

    Calibration fitted on all candidates can be right on average yet overconfident in the
    extreme tail where selection happens. Compare predicted vs realized on the chosen set."""
    d = [c for c in chosen if c["outcome"] in ("win", "loss")]
    if not d:
        return None
    from v13.pricing import profit_per_unit
    hit = sum(c["outcome"] == "win" for c in d) / len(d)
    roi = sum(profit_per_unit(c["price"]) if c["outcome"] == "win" else -1 for c in d) / len(d)
    p_hat = sum(c["p"] for c in d) / len(d)
    p_mkt = [c["p_market"] for c in d if c.get("p_market") is not None]
    return {"n": len(d), "predicted_hit": round(p_hat, 4), "realized_hit": round(hit, 4),
            "predicted_ev_pct": round(100 * sum(c["ev"] for c in d) / len(d), 2), "realized_roi_pct": round(100 * roi, 2),
            "market_hit": round(sum(p_mkt) / len(p_mkt), 4) if p_mkt else None,
            "realized_edge_share": round((hit - sum(p_mkt) / len(p_mkt)) / (p_hat - sum(p_mkt) / len(p_mkt)), 3)
            if p_mkt and abs(p_hat - sum(p_mkt) / len(p_mkt)) > 1e-6 else None,
            "reading_guide": "realized_edge_share ~1 = edge as predicted; ~0 = none of the predicted edge was real; "
                             "use it as the shrink factor in the sensitivity block."}

TARGETS = (100, 150, 200, 250, 300)


def rank_buckets(val, test, edges=(0.01, 0.05, 0.20)):
    """Does a higher rank mean a better bet? EV_lo cut points are fixed on VALIDATION quantiles,
    then test candidates (decided, with market price) are bucketed and graded."""
    import math
    from v13.pricing import profit_per_unit
    v = sorted((c["ev_lo"] for c in val), reverse=True)
    if not v:
        return None
    cuts = [v[min(len(v) - 1, int(len(v) * e))] for e in edges]
    names = [f"top_{int(100 * edges[0])}pct"] + [f"{int(100 * a)}-{int(100 * b)}pct" for a, b in zip(edges, edges[1:])] + ["rest"]
    buckets = {n: [] for n in names}
    for c in test:
        if c["outcome"] not in ("win", "loss") or c.get("p_market") is None:
            continue
        i = next((k for k, cut in enumerate(cuts) if c["ev_lo"] >= cut), len(cuts))
        buckets[names[i]].append(c)
    out = {"validation_cutpoints_ev_lo": [round(x, 5) for x in cuts]}
    for n, rows in buckets.items():
        if not rows:
            out[n] = {"n": 0}; continue
        w = sum(c["outcome"] == "win" for c in rows); exp = sum(c["p_market"] for c in rows)
        var = sum(c["p_market"] * (1 - c["p_market"]) for c in rows)
        out[n] = {"n": len(rows), "mean_p": round(sum(c["p"] for c in rows) / len(rows), 4),
                  "realized_hit": round(w / len(rows), 4), "market_hit": round(exp / len(rows), 4),
                  "excess": round((w - exp) / len(rows), 4), "z": round((w - exp) / math.sqrt(var), 2) if var else None,
                  "roi_pct": round(100 * sum(profit_per_unit(c["price"]) if c["outcome"] == "win" else -1 for c in rows) / len(rows), 2)}
    return out


def game_cluster_bootstrap_roi(bets, reps=2000, seed=5):
    """95% CI for flat ROI resampling whole GAMES (correlated props move together)."""
    import numpy as np
    from v13.pricing import profit_per_unit
    by = {}
    for b in bets:
        if b["outcome"] in ("win", "loss"):
            by.setdefault(str(b["game_id"]), []).append(profit_per_unit(b["price"]) if b["outcome"] == "win" else -1.0)
    groups = list(by.values())
    if len(groups) < 10:
        return None
    rng = np.random.default_rng(seed)
    rois = []
    for _ in range(reps):
        pick = [groups[i] for i in rng.integers(0, len(groups), len(groups))]
        flat = [x for g in pick for x in g]
        rois.append(100 * sum(flat) / len(flat))
    return {"games": len(groups), "roi_95ci_pct": [round(float(np.percentile(rois, 2.5)), 2), round(float(np.percentile(rois, 97.5)), 2)]}


def read(path):
    return [json.loads(x) for x in Path(path).open() if x.strip()]


def run(candidates, validation_season=None, test_season=None, models=None, targets=TARGETS, mc_paths=10000,
        split_date=None, grouping="market"):
    """Seasons mode (preferred) or, when only one season has lines, split_date mode:
    validation = scored candidates before split_date, test = on/after it (counts scale with window length)."""
    if models:
        candidates = [c for c in candidates if c.get("model_version") in models]
    scored, fits = walk_forward_score(candidates, grouping=grouping)
    if split_date:
        pool = [c for c in scored if test_season is None or c.get("season") == test_season]
        val = [c for c in pool if c["date"] < split_date]
        test = [c for c in pool if c["date"] >= split_date]
    else:
        val = [c for c in scored if c.get("season") == validation_season]
        test = [c for c in scored if c.get("season") == test_season]
    if val and test and max(c["date"] for c in val) >= min(c["date"] for c in test):
        raise AssertionError("validation window overlaps test window")
    if val and test and {c["candidate_id"] for c in val} & {c["candidate_id"] for c in test}:
        raise AssertionError("candidate appears in both validation and test")
    if not val or not test:
        # Never relabel seasons or borrow test months: report the gap and stop cleanly.
        return {"status": "UNTESTABLE", "grouping": grouping,
                "reason": f"no scored candidates in the {'validation' if not val else 'test'} window",
                "validation_season": validation_season, "test_season": test_season, "split_date": split_date,
                "raw_candidates_by_season": dict(Counter(str(c.get("season")) for c in candidates)),
                "scored_candidates_by_season": dict(Counter(str(c.get("season")) for c in scored)),
                "scored_validation": len(val), "scored_test": len(test)}
    scale = 1.0
    if split_date:
        days = lambda rows: len({c["date"] for c in rows})
        scale = days(val) / max(days(test), 1)
    classes = {c.get("evidence_class", "exploratory_unverified_prices") for c in candidates}
    report = {"evidence_class": "validated" if classes == {"validated"} else "exploratory_unverified_prices",
              "candidate_evidence_classes": sorted(classes),
              "grouping": grouping, "validation_season": validation_season, "test_season": test_season, "split_date": split_date,
              "validation_to_test_day_ratio": round(scale, 3),
              "scored_candidates": dict(Counter(c["market"] for c in scored)),
              "calibration_fits_last_period": fits[max(fits)],
              "reliability_test": {m: reliability([c for c in test if c["market"] == m])
                                   for m in sorted({c["market"] for c in test})},
              "rank_buckets_test": rank_buckets(val, test),
              "sweep": {}}
    for n in targets:
        tau = tau_for_target(val, max(1, round(n * scale)) if split_date else n)
        chosen = select(test, tau)
        sims = {st: simulate(chosen, st) for st in STAKING}
        flat = sims["flat"]
        report["sweep"][str(n)] = {
            "ev_lo_cutoff_from_validation": round(tau, 5),
            "validation_count": len(select(val, tau)),
            "test": {**{f"{st}": {k: v for k, v in r.items() if k != "ledger"} for st, r in sims.items()},
                     **{f"monte_carlo_{st}": monte_carlo(r["ledger"], st, mc_paths) for st, r in sims.items()}},
            "actionable_qualified_in_test": len(chosen),
            "tier_validation": __import__("v13.tier_validation", fromlist=["validate_tiers"]).validate_tiers(chosen),
            "shortfall_vs_target": max(0, n - len(chosen)),
            "game_cluster_roi_ci": game_cluster_bootstrap_roi(chosen),
            "selected_bets_calibration_test": selection_calibration(chosen),
            "selected_bets_calibration_validation": selection_calibration(select(val, tau)),
            "sensitivity": {f"{st}_shrink_{k}": parametric_monte_carlo(chosen, k, st, max(1000, mc_paths // 2))
                            for st in ("flat", "tiered") for k in SHRINKS},
            "test_ledger_flat": flat["ledger"]}
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--candidates", action="append", required=True)
    p.add_argument("--models", help="comma-separated model_version values to include")
    p.add_argument("--validation-season", type=int)
    p.add_argument("--test-season", type=int)
    p.add_argument("--split-date", help="YYYY-MM-DD; single-season mode (weaker, exploratory)")
    p.add_argument("--mc-paths", type=int, default=10000)
    p.add_argument("--grouping", choices=("market", "pooled"), default="market")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    cands = [c for path in a.candidates for c in read(path)]
    models = set(a.models.split(",")) if a.models else None
    if not a.split_date and (a.validation_season is None or a.test_season is None):
        raise SystemExit("Give --validation-season and --test-season, or --split-date")
    report = run(cands, a.validation_season, a.test_season, models, mc_paths=a.mc_paths, split_date=a.split_date,
                 grouping=a.grouping)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str) + "\n")
    if report.get("status") == "UNTESTABLE":
        print(json.dumps(report, indent=2))
        print("STATUS: UNTESTABLE (no relabelling; see reason)")
        return
    brief = {k: v for k, v in report.items() if k != "sweep"}
    brief["sweep"] = {n: {"cutoff": s["ev_lo_cutoff_from_validation"], "validation_count": s["validation_count"],
                          **{k: {kk: vv for kk, vv in s["test"][k].items() if kk not in ("by_tier", "by_market")}
                             for k in STAKING}} for n, s in report["sweep"].items()}
    print(json.dumps(brief, indent=2, default=str))


if __name__ == "__main__":
    main()
