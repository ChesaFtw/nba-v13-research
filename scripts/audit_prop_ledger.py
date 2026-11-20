"""Integrity audit of an existing prop holdout ledger against its source lines.

Answers: does the hit rate survive (1) real opening prices, (2) a strict
pre-tip timestamp check, (3) duplicate/alternate line checks, and (4) the
market's own de-vigged expectation? Run it on the artifacts you already have:

python -m scripts.audit_prop_ledger \
  --report research_results/prop_backtest/holdout_2025.json \
  --lines research_results/prop_openings_full/opening_props_2025.jsonl \
  --games research_data/game_history/games_2025.jsonl \
  --market threes --out research_results/integrity/threes_ledger_audit_2025.json

Descriptive only. Never use these slices to pick new thresholds on the same holdout.
"""
import argparse, json, math
from collections import Counter, defaultdict
from pathlib import Path

from scripts.backtest_prop_holdout import ALIASES, normalize_lines
from v13.pricing import (ASSUMED_110, implied_probability, no_vig_pair, parse_utc, pregame_status,
                         profit_per_unit, prop_prices, tip_times, wilson)


def read_jsonl(path):
    return [json.loads(x) for x in Path(path).open() if x.strip()]


def line_bucket(line):
    if line <= 0.5: return "0.5"
    if line <= 1.5: return "1.5"
    if line <= 2.5: return "2.5"
    if line <= 3.5: return "3.5"
    return "4.5+"


def price_bucket(odds):
    if odds is None: return "missing"
    if odds <= -150: return "<=-150"
    if odds <= -121: return "-149..-121"
    if odds <= -100: return "-120..-100"
    return "plus_money"


def timing_bucket(status, minutes):
    if status != "pregame": return status
    if minutes < 60: return "pregame_<1h"
    if minutes < 360: return "pregame_1-6h"
    return "pregame_>6h"


def summarize(bets):
    decided = [b for b in bets if b["outcome"] in ("win", "loss")]
    n = len(decided); w = sum(b["outcome"] == "win" for b in decided)
    out = {"bets": len(bets), "wins": w, "losses": n - w,
           "pushes": len(bets) - n, "hit_rate": round(w / n, 4) if n else None,
           "hit_rate_95ci": wilson(w, n)}
    out["roi_flat_at_assumed_110_pct"] = round(100 * sum(
        profit_per_unit(ASSUMED_110) if b["outcome"] == "win" else -1 for b in decided) / n, 2) if n else None
    priced = [b for b in decided if b["price"] is not None]
    out["priced_bets"] = len(priced)
    if priced:
        out["roi_flat_at_actual_price_pct"] = round(100 * sum(
            profit_per_unit(b["price"]) if b["outcome"] == "win" else -1 for b in priced) / len(priced), 2)
        out["mean_breakeven_hit_rate"] = round(sum(implied_probability(b["price"]) for b in priced) / len(priced), 4)
    fair = [b for b in priced if b["market_prob"] is not None]
    if fair:
        # Market-adjusted edge: realized wins minus what the de-vigged market expected.
        exp = sum(b["market_prob"] for b in fair)
        var = sum(b["market_prob"] * (1 - b["market_prob"]) for b in fair)
        wins = sum(b["outcome"] == "win" for b in fair)
        out["market_expected_hit_rate"] = round(exp / len(fair), 4)
        out["excess_hit_rate_vs_market"] = round((wins - exp) / len(fair), 4)
        out["excess_z_score"] = round((wins - exp) / math.sqrt(var), 2) if var > 0 else None
    return out


def audit(report, lines, games, market):
    ledger = (report.get("markets", {}).get(market) or {}).get("ridge_holdout_ledger") or []
    if not ledger:
        raise RuntimeError(f"No holdout ledger for market '{market}' in report")
    chosen = normalize_lines(lines)  # exactly the row the backtest graded
    tips = tip_times(games)

    all_rows = defaultdict(list)
    for r in lines:
        m = ALIASES.get(r.get("prop_type"))
        if m == market and (r.get("market") or {}).get("type") == "over_under":
            all_rows[(str(r["game_id"]), str(r["player_id"]))].append(r)
    multi_rows = sum(len(v) > 1 for v in all_rows.values())
    multi_values = sum(len({str(x.get("line_value")) for x in v}) > 1 for v in all_rows.values())
    same_vendor_multi = sum(any(c > 1 for c in Counter(str(x.get("vendor")).lower() for x in v).values())
                            for v in all_rows.values())

    bets, unmatched = [], 0
    for b in ledger:
        found = chosen.get((str(b["game_id"]), str(b["player_id"]), market))
        if found is None or abs(float(found[1]) - float(b["line"])) > 1e-9:
            unmatched += 1; continue
        row = found[0]
        over, under = prop_prices(row)
        price = over if b["side"] == "over" else under
        market_prob = None
        if over is not None and under is not None:
            p_over, p_under = no_vig_pair(over, under)
            market_prob = p_over if b["side"] == "over" else p_under
        tip = tips.get(str(b["game_id"]))
        status = pregame_status(row.get("opened_at"), tip)
        minutes = None
        if status == "pregame":
            minutes = (tip - parse_utc(row["opened_at"])).total_seconds() / 60
        edge = abs(float(b["projection"]) - float(b["line"]))
        bets.append({**b, "price": price, "market_prob": market_prob, "timing": timing_bucket(status, minutes),
                     "line_bucket": line_bucket(float(b["line"])), "price_bucket": price_bucket(price),
                     "edge_bucket": "<1" if edge < 1 else ("1-1.5" if edge < 1.5 else ">=1.5"),
                     "month": str(b["date"])[:7], "vendor": str(row.get("vendor"))})

    def by(field):
        groups = defaultdict(list)
        for x in bets: groups[x[field]].append(x)
        return {k: summarize(v) for k, v in sorted(groups.items())}

    return {
        "market": market,
        "ledger_bets": len(ledger), "matched_to_source_row": len(bets), "unmatched": unmatched,
        "games_with_tip_time": len(tips),
        "source_keys": len(all_rows),
        "keys_with_multiple_rows": multi_rows,
        "keys_with_multiple_distinct_line_values": multi_values,
        "keys_with_multiple_rows_same_vendor": same_vendor_multi,
        "overall": summarize(bets),
        "pregame_verified_only": summarize([x for x in bets if x["timing"].startswith("pregame")]),
        "by_side": by("side"), "by_line": by("line_bucket"), "by_price": by("price_bucket"),
        "by_timing": by("timing"), "by_vendor": by("vendor"), "by_month": by("month"),
        "by_edge": by("edge_bucket"),
        "reading_guide": [
            "If roi_flat_at_actual_price is far below the -110 ROI, the edge was mostly vig mis-pricing.",
            "excess_hit_rate_vs_market is the honest edge estimate: realized wins minus de-vigged market expectation.",
            "Any post_tip or unverifiable rows invalidate those bets; judge on pregame_verified_only.",
            "keys_with_multiple_rows_same_vendor > 0 means the earliest-row tiebreak picked among several lines; inspect them.",
        ],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--report", required=True); p.add_argument("--lines", required=True)
    p.add_argument("--games", required=True); p.add_argument("--market", default="threes")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    result = audit(json.loads(Path(a.report).read_text()), read_jsonl(a.lines), read_jsonl(a.games), a.market)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if not k.startswith("by_")}, indent=2))


if __name__ == "__main__":
    main()
