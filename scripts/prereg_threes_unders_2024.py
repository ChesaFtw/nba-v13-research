"""Experiment A: frozen 2024-25 three-point UNDERS replication (PREREG_THREES_UNDERS.md).

Selection is the LEGACY pipeline, unchanged: scripts.backtest_prop_holdout.evaluate
(same normalize_lines, features, Ridge(alpha=25), scaling, 65/35 date split, threshold grid
[0.5, 1.0, 1.5], train-only threshold choice). Bets are then graded, never re-selected:
  - only bets whose source line was opened strictly before tip are counted;
  - each bet is paid at its recorded side-specific opening price;
  - fair probability under BOTH multiplicative and power de-vig.
The frozen pass rule is applied mechanically and written into the output.

python -m scripts.prereg_threes_unders_2024 --features props_2024.jsonl \
  --lines opening_threes_2024.jsonl --games games_2024.jsonl --out research_results/prereg/threes_unders_2024.json
"""
import argparse, json, math
from collections import Counter
from pathlib import Path

from scripts.backtest_prop_holdout import evaluate as legacy_evaluate, normalize_lines
from v13.pricing import (no_vig_pair, power_devig_pair, pregame_status, profit_per_unit, prop_prices,
                         tip_times, wilson)

PASS_RULE = {"min_unders": 150, "min_roi": 0.0, "min_excess": 0.02, "min_z": 2.0}


def read(path):
    return [json.loads(x) for x in Path(path).open() if x.strip()]


def grade(bets):
    decided = [b for b in bets if b["outcome"] in ("win", "loss")]
    n = len(decided); w = sum(b["outcome"] == "win" for b in decided)
    out = {"bets": len(bets), "decided": n, "wins": w, "losses": n - w, "pushes": len(bets) - n,
           "hit_rate": round(w / n, 4) if n else None, "hit_95ci": wilson(w, n)}
    if not n:
        return out
    pnl = sum(profit_per_unit(b["price"]) if b["outcome"] == "win" else -1 for b in decided)
    out["roi_actual_price_pct"] = round(100 * pnl / n, 2)
    for name, key in (("multiplicative", "p_mult"), ("power", "p_power")):
        exp = sum(b[key] for b in decided); var = sum(b[key] * (1 - b[key]) for b in decided)
        out[name] = {"expected_hit_rate": round(exp / n, 4), "excess": round((w - exp) / n, 4),
                     "z": round((w - exp) / math.sqrt(var), 2) if var > 0 else None}
    return out


def run(features, lines, games):
    report = legacy_evaluate(features, lines)  # selection exactly as frozen
    threes = report["markets"].get("threes") or {}
    if threes.get("status") != "evaluated":
        return {"verdict": "UNTESTABLE", "reason": f"legacy pipeline did not evaluate threes: {threes}"}
    chosen = normalize_lines(lines)
    tips = tip_times(games)
    integrity, bets = Counter(), []
    for b in threes["ridge_holdout_ledger"]:
        found = chosen.get((str(b["game_id"]), str(b["player_id"]), "threes"))
        if found is None or abs(float(found[1]) - float(b["line"])) > 1e-9:
            integrity["unmatched_source_row"] += 1; continue
        row = found[0]
        status = pregame_status(row.get("opened_at"), tips.get(str(b["game_id"])))
        if status != "pregame":
            integrity[f"excluded_{status}"] += 1; continue
        over, under = prop_prices(row)
        if over is None or under is None:
            integrity["excluded_missing_price"] += 1; continue
        mo, mu = no_vig_pair(over, under); po, pu = power_devig_pair(over, under)
        side = b["side"]
        bets.append({**b, "price": over if side == "over" else under,
                     "p_mult": mo if side == "over" else mu, "p_power": po if side == "over" else pu})
        integrity["graded"] += 1
    unders = [b for b in bets if b["side"] == "under"]
    primary = grade(unders)
    checks = {
        "unders_n_ge_150": primary["decided"] >= PASS_RULE["min_unders"],
        "roi_gt_0": (primary.get("roi_actual_price_pct") or -1) > PASS_RULE["min_roi"],
        "excess_ge_2pts_multiplicative": (primary.get("multiplicative") or {}).get("excess", -1) >= PASS_RULE["min_excess"],
        "excess_ge_2pts_power": (primary.get("power") or {}).get("excess", -1) >= PASS_RULE["min_excess"],
        "z_ge_2_multiplicative": ((primary.get("multiplicative") or {}).get("z") or -9) >= PASS_RULE["min_z"],
        "z_ge_2_power": ((primary.get("power") or {}).get("z") or -9) >= PASS_RULE["min_z"],
    }
    verified_share = integrity["graded"] / max(1, len(threes["ridge_holdout_ledger"]))
    if verified_share < 0.9:
        verdict = "UNTESTABLE"  # pre-registration: insufficient integrity is not a pass
    else:
        verdict = "PASS" if all(checks.values()) else "FAIL"
    return {"experiment": "A_threes_unders_2024_25", "preregistration": "PREREG_THREES_UNDERS.md",
            "verdict": verdict, "checks": checks, "pass_rule": PASS_RULE,
            "selection": {"holdout_start": report["holdout_start"], "chosen_threshold": threes["chosen_edge_threshold"],
                          "legacy_holdout_record_at_-110": threes["ridge_holdout"]},
            "integrity": dict(integrity), "verified_share": round(verified_share, 4),
            "primary_unders": primary, "secondary_overs": grade([b for b in bets if b["side"] == "over"]),
            "secondary_all": grade(bets),
            "descriptive_only_by_price": {f"{lo}..{hi}": grade([b for b in unders if lo <= b["price"] < hi])
                                          for lo, hi in ((-10000, -200), (-200, -150), (-150, -100), (100, 10000))}}


def main():
    p = argparse.ArgumentParser()
    for k in ("features", "lines", "games", "out"):
        p.add_argument(f"--{k}", required=True)
    a = p.parse_args()
    result = run(read(a.features), read(a.lines), read(a.games))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "descriptive_only_by_price"}, indent=2))
    print(f"VERDICT: {result['verdict']}")


if __name__ == "__main__":
    main()
