"""Do confidence tiers mean anything? One report for backtest ledgers and the prospective ledger.

Input: bets with tier, price, outcome, p_market (de-vigged, optional), date, game_id.
Per tier: n, hit rate (Wilson CI), mean break-even, excess over market with z, flat ROI.
Ordering checks A < A+ < A++ on excess and ROI, each requiring >= min_n decided bets per tier.
Also: qualified bets per month, and correlated (same-game multi-bet) vs single-bet results.
"""
import math
from collections import Counter, defaultdict

from v13.pricing import implied_probability, profit_per_unit, wilson

ORDER = ("A", "A+", "A++")


def _grade(rows):
    d = [b for b in rows if b["outcome"] in ("win", "loss")]
    n = len(d)
    out = {"bets": len(rows), "decided": n}
    if not n:
        return out
    w = sum(b["outcome"] == "win" for b in d)
    out.update(hit_rate=round(w / n, 4), hit_95ci=wilson(w, n),
               mean_breakeven=round(sum(implied_probability(b["price"]) for b in d) / n, 4),
               roi_pct=round(100 * sum(profit_per_unit(b["price"]) if b["outcome"] == "win" else -1 for b in d) / n, 2))
    m = [b for b in d if b.get("p_market") is not None]
    if m:
        exp = sum(b["p_market"] for b in m); var = sum(b["p_market"] * (1 - b["p_market"]) for b in m)
        wm = sum(b["outcome"] == "win" for b in m)
        out.update(excess=round((wm - exp) / len(m), 4), z=round((wm - exp) / math.sqrt(var), 2) if var else None)
    return out


def validate_tiers(bets, min_n=100):
    tiers = {t: _grade([b for b in bets if b.get("tier") == t]) for t in ORDER}
    stray = [b for b in bets if b.get("tier") not in ORDER]
    ordered = {}
    for key in ("excess", "roi_pct"):
        vals = [tiers[t].get(key) for t in ORDER]
        enough = all(tiers[t]["decided"] >= min_n for t in ORDER)
        ordered[key] = ("insufficient_sample" if not enough else
                        "increasing" if all(v is not None for v in vals) and vals[0] < vals[1] < vals[2] else "not_increasing")
    per_game = Counter(str(b["game_id"]) for b in bets)
    corr = {"single_bet_games": _grade([b for b in bets if per_game[str(b["game_id"])] == 1]),
            "multi_bet_games": _grade([b for b in bets if per_game[str(b["game_id"])] > 1])}
    monthly = Counter(str(b["date"])[:7] for b in bets if b.get("tier") in ORDER)
    verdict = ("tiers_supported" if ordered["excess"] == ordered["roi_pct"] == "increasing"
               else "use_flat_staking")
    return {"by_tier": tiers, "ordering": ordered, "min_n_per_tier": min_n, "verdict": verdict,
            "non_actionable_rows_present": len(stray), "qualified_per_month": dict(sorted(monthly.items())),
            "correlation": corr, "all": _grade([b for b in bets if b.get("tier") in ORDER])}
