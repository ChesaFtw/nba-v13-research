"""Frozen procedure replication for one prop market on an untouched season (v2 pipeline).

Runs scripts.backtest_prop_holdout_v2.evaluate EXACTLY as configured in code (within-season
65/35 split, inner 70/30 validation, EDGE_GRID/EV_GRID, >=40 validation bets, actual prices,
strict pre-tip lines), takes the named market + selector holdout ledger, and grades it under
multiplicative AND power de-vig with a mechanical verdict. Nothing is chosen by this script.

python -m scripts.prereg_v2_market --market rebounds --selector edge --features props_2024.jsonl \
  --lines opening_props_2024.jsonl --games games_2024.jsonl --out research_results/prereg/rebounds_2024.json
"""
import argparse, json, math
from pathlib import Path

from scripts.backtest_prop_holdout_v2 import evaluate
from v13.pricing import no_vig_pair, power_devig_pair, profit_per_unit, wilson

RULE = {"min_bets": 150, "min_roi_pct": 0.0, "min_excess": 0.02, "min_z": 2.0, "min_priced_share": 0.9}


def read(p):
    return [json.loads(x) for x in Path(p).open() if x.strip()]


def grade(ledger):
    d = [b for b in ledger if b["outcome"] in ("win", "loss")]
    n = len(d); w = sum(b["outcome"] == "win" for b in d)
    out = {"bets": len(ledger), "decided": n, "wins": w, "losses": n - w, "pushes": len(ledger) - n,
           "hit_rate": round(w / n, 4) if n else None, "hit_95ci": wilson(w, n)}
    if not n:
        return out
    out["roi_actual_price_pct"] = round(100 * sum(profit_per_unit(b["price"]) if b["outcome"] == "win" else -1
                                                  for b in d) / n, 2)
    for name, fn in (("multiplicative", no_vig_pair), ("power", power_devig_pair)):
        ps = [fn(b["over_odds"], b["under_odds"])[0 if b["side"] == "over" else 1] for b in d]
        exp, var = sum(ps), sum(p * (1 - p) for p in ps)
        out[name] = {"expected_hit_rate": round(exp / n, 4), "excess": round((w - exp) / n, 4),
                     "z": round((w - exp) / math.sqrt(var), 2) if var > 0 else None}
    return out


def run(features, lines, games, market, selector):
    rep = evaluate(features, lines, games, "actual")
    m = rep["markets"].get(market) or {}
    if m.get("status") != "evaluated" or not isinstance(m.get(selector), dict) or "holdout_ledger" not in m[selector]:
        return {"verdict": "UNTESTABLE", "reason": f"pipeline did not evaluate {market}/{selector}", "detail": m,
                "coverage": rep.get("coverage")}
    ledger = m[selector]["holdout_ledger"]
    cov = rep.get("coverage", {})
    dropped = cov.get("line_post_tip", 0) + cov.get("line_unverifiable", 0)
    priced_share = rep.get("rows_with_both_prices", 0) / max(1, rep.get("matched_rows", 0))
    primary = grade(ledger)
    checks = {
        "bets_ge_150": primary["decided"] >= RULE["min_bets"],
        "roi_gt_0": (primary.get("roi_actual_price_pct") is not None and primary["roi_actual_price_pct"] > RULE["min_roi_pct"]),
        "excess_ge_2pts_multiplicative": (primary.get("multiplicative") or {}).get("excess", -1) >= RULE["min_excess"],
        "excess_ge_2pts_power": (primary.get("power") or {}).get("excess", -1) >= RULE["min_excess"],
        "z_ge_2_multiplicative": ((primary.get("multiplicative") or {}).get("z") or -9) >= RULE["min_z"],
        "z_ge_2_power": ((primary.get("power") or {}).get("z") or -9) >= RULE["min_z"],
    }
    verdict = "UNTESTABLE" if priced_share < RULE["min_priced_share"] else ("PASS" if all(checks.values()) else "FAIL")
    return {"verdict": verdict, "market": market, "selector": selector, "rule": RULE, "checks": checks,
            "frozen_threshold": m[selector]["threshold"], "holdout_start": rep.get("holdout_start"),
            "priced_share": round(priced_share, 4), "lines_dropped_not_pregame": dropped, "coverage": cov,
            "primary": primary,
            "secondary_by_side": {s: grade([b for b in ledger if b["side"] == s]) for s in ("over", "under")},
            "validation_block": m[selector]["validation"]}


def main():
    p = argparse.ArgumentParser()
    for k in ("market", "selector", "features", "lines", "games", "out"):
        p.add_argument(f"--{k}", required=True)
    a = p.parse_args()
    res = run(read(a.features), read(a.lines), read(a.games), a.market, a.selector)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=2))
    print(f"VERDICT: {res['verdict']}")


if __name__ == "__main__":
    main()
