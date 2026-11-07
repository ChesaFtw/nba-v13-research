"""How much forecasting edge does $10k -> $15k actually require?

For a bet count N, a price profile and a staking plan, find the excess win probability over
the de-vigged fair probability (in percentage points) at which the median season ends at
$15,000, and report outcome distributions across a grid of edges. Pure simulation: it uses
no historical results, so it can be run before any model exists.

python -m scripts.bankroll_feasibility --out research_results/feasibility.json
"""
import argparse, json

import numpy as np

from v13.pricing import implied_probability, profit_per_unit

PROFILES = {
    "minus110_both_sides": [(-110, -110)],
    "favourite_unders_-178": [(-178, 140)],
    "plus_money_+130": [(130, -160)],
    "mixed_props": [(-110, -110), (-130, 105), (-150, 120), (-178, 140), (115, -140)],
}
PLANS = {"flat_1pct": [(0.01, 1.0)], "flat_2pct": [(0.02, 1.0)],
         "tiers_1_2_3_mix_60_30_10": [(0.01, 0.6), (0.02, 0.3), (0.03, 0.1)],
         "tiers_capped_2pct": [(0.01, 0.6), (0.02, 0.4)]}


def fair_prob(price, opp):
    a, b = implied_probability(price), implied_probability(opp)
    return a / (a + b)


def simulate(n, profile, plan, edge, paths=20000, seed=3, per_day=3):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(profile), (paths, n))
    fair = np.array([fair_prob(p, o) for p, o in profile])[idx]
    payout = np.array([profit_per_unit(p) for p, _ in profile])[idx]
    win = rng.random((paths, n)) < np.clip(fair + edge, 0, 1)
    fr, wts = zip(*plan)
    stake = rng.choice(fr, size=(paths, n), p=np.array(wts) / sum(wts))
    ret = np.where(win, payout, -1.0) * stake
    bank = np.full(paths, 10000.0); peak = bank.copy(); mdd = np.zeros(paths)
    for start in range(0, n, per_day):  # bets in groups of `per_day` share a bankroll snapshot
        snap = bank.copy()
        bank = bank + snap * ret[:, start:start + per_day].sum(axis=1)
        peak = np.maximum(peak, bank); mdd = np.maximum(mdd, (peak - bank) / peak)
    roi = float((ret.sum(axis=1) / stake.sum(axis=1)).mean())
    return {"edge_pts": round(100 * edge, 2), "mean_roi_pct": round(100 * roi, 2),
            "median_end": round(float(np.median(bank))), "prob_ge_15000": round(float((bank >= 15000).mean()), 3),
            "prob_below_10000": round(float((bank < 10000).mean()), 3),
            "median_max_dd_pct": round(100 * float(np.median(mdd)), 1),
            "p95_max_dd_pct": round(100 * float(np.percentile(mdd, 95)), 1)}


def required_edge(n, profile, plan, target_prob=0.5, paths=8000):
    lo, hi = 0.0, 0.25
    for _ in range(12):
        mid = (lo + hi) / 2
        if simulate(n, profile, plan, mid, paths)["prob_ge_15000"] >= target_prob:
            hi = mid
        else:
            lo = mid
    return round(100 * hi, 2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--counts", default="100,150,200,250,300")
    a = p.parse_args()
    counts = [int(x) for x in a.counts.split(",")]
    report = {"note": "edge = win probability above the de-vigged fair probability; 3 bets share each bankroll snapshot",
              "required_edge_for_median_15k": {}, "grid_200_bets": {}}
    for prof_name, prof in PROFILES.items():
        for plan_name, plan in PLANS.items():
            key = f"{prof_name}|{plan_name}"
            report["required_edge_for_median_15k"][key] = {str(n): required_edge(n, prof, plan) for n in counts}
            report["grid_200_bets"][key] = [simulate(200, prof, plan, e) for e in (0.0, 0.02, 0.03, 0.05, 0.08)]
    out = __import__("pathlib").Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["required_edge_for_median_15k"], indent=2))


if __name__ == "__main__":
    main()
