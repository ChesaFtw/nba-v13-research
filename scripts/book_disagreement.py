"""R12 (pre-registered in PREREG_R12_BOOK_DISAGREEMENT.md): when one sportsbook's opening prop
disagrees with the other books, does betting against that book beat its price?

No projection of our own is used. The other books are the model.

H12a PRICE OUTLIER (same line). For a book v, the "others" are books quoting the SAME line for the
same player-game-market whose opening timestamp is within WINDOW_MIN of v's (so the comparison is
roughly simultaneous). Fair P(over) = mean of the others' power-de-vigged P(over). The bet is the
side of v whose expected value at v's actual price is >= MIN_EV under that fair probability.
H12b LINE OUTLIER. v's line differs from the median line of the in-window others by >= MIN_LINE_GAP.
Bet the side v made easier (over if v's line is lower, under if higher), only if that side's price
is no worse than MAX_JUICE.

One bet per player-game-market per hypothesis (the largest EV / largest gap; ties by vendor
priority). DNP props are void (no box score); pushes are excluded from ROI. Inputs must be
provenance-gated. Frozen constants below; nothing is tuned here.
"""
import argparse, json, math, statistics
from collections import Counter, defaultdict
from datetime import timedelta
from pathlib import Path

import numpy as np

from scripts.backtest_prop_holdout import ALIASES, VENDORS
from v13.pricing import parse_utc, power_devig_pair, no_vig_pair, profit_per_unit, prop_prices, wilson
from v13.provenance import require_eligible

FROZEN = {"markets": ("points", "rebounds", "assists", "threes", "pra"), "min_books": 3, "min_others": 2,
          "window_min": 60, "min_ev": 0.03, "min_line_gap": 1.0, "max_juice": -150,
          "pass": {"min_bets": 300, "min_roi_pct": 0.0, "min_z": 2.5, "min_cluster_ci_low_pct": 0.0}}
STAT = {"points": ("pts",), "rebounds": ("reb",), "assists": ("ast",), "threes": ("fg3m",), "pra": ("pts", "reb", "ast")}
CENTRAL = timedelta(hours=-5)  # descriptive only: hour the outlier price was posted, US Central (CDT/CST approx.)


def read(p):
    return [json.loads(x) for x in Path(p).open() if x.strip()]


def vrank(v):
    return VENDORS.index(v) if v in VENDORS else 99


def opening_quotes(props):
    """(game, player, market) -> {vendor: quote}; the earliest gated two-sided row per vendor."""
    best = {}
    for r in props:
        m = ALIASES.get(r.get("prop_type"))
        if m not in FROZEN["markets"] or (r.get("market") or {}).get("type") != "over_under":
            continue
        over, under = prop_prices(r)
        if over is None or under is None or r.get("line_value") is None or not r.get("opened_at"):
            continue
        v = str(r.get("vendor") or "").lower()
        key = (str(r["game_id"]), str(r["player_id"]), m, v)
        t = parse_utc(r["opened_at"])
        if key not in best or t < best[key]["t"]:
            best[key] = {"vendor": v, "line": float(r["line_value"]), "over": over, "under": under, "t": t,
                         "p_over": power_devig_pair(over, under)[0]}
    groups = defaultdict(dict)
    for (g, p, m, v), q in best.items():
        groups[(g, p, m)][v] = q
    return groups


def in_window(a, b):
    return abs((a["t"] - b["t"]).total_seconds()) <= FROZEN["window_min"] * 60


def pick_bets(groups):
    price_bets, line_bets, seen = [], [], Counter()
    for (gid, pid, m), books in groups.items():
        seen["groups"] += 1
        if len(books) < FROZEN["min_books"]:
            seen["fewer_than_3_books"] += 1; continue
        seen["groups_ge_3_books"] += 1
        best_price, best_line = None, None
        for v, q in books.items():
            near = [o for w, o in books.items() if w != v and in_window(q, o)]
            same = [o for o in near if abs(o["line"] - q["line"]) < 1e-9]
            if len(same) >= FROZEN["min_others"]:
                fair_over = sum(o["p_over"] for o in same) / len(same)
                for side, fair, price in (("over", fair_over, q["over"]), ("under", 1 - fair_over, q["under"])):
                    ev = fair * (1 + profit_per_unit(price)) - 1
                    if ev >= FROZEN["min_ev"]:
                        cand = (ev, -vrank(v), {"vendor": v, "side": side, "line": q["line"], "price": price,
                                                "opp_price": q["under"] if side == "over" else q["over"],
                                                "fair": fair, "ev": ev, "others": len(same), "t": q["t"]})
                        if best_price is None or cand[:2] > best_price[:2]:
                            best_price = cand
            if len(near) >= FROZEN["min_others"]:
                med = statistics.median(o["line"] for o in near)
                gap = med - q["line"]
                if abs(gap) >= FROZEN["min_line_gap"]:
                    side = "over" if gap > 0 else "under"
                    price = q["over"] if side == "over" else q["under"]
                    if price is not None and (price > 0 or price >= FROZEN["max_juice"]):
                        cand = (abs(gap), -vrank(v), {"vendor": v, "side": side, "line": q["line"], "price": price,
                                                      "opp_price": q["under"] if side == "over" else q["over"],
                                                      "gap": gap, "others": len(near), "t": q["t"]})
                        if best_line is None or cand[:2] > best_line[:2]:
                            best_line = cand
        for best, out in ((best_price, price_bets), (best_line, line_bets)):
            if best:
                out.append({"game_id": gid, "player_id": pid, "market": m, **best[2]})
    return price_bets, line_bets, seen


def settle(bets, stats):
    box = {}
    for r in stats:
        g, pl = r.get("game") or {}, r.get("player") or {}
        if g.get("id") is not None and pl.get("id") is not None:
            box[(str(g["id"]), str(pl["id"]))] = r
    out, skipped = [], Counter()
    for b in bets:
        r = box.get((b["game_id"], b["player_id"]))
        if r is None:
            skipped["no_box_score_void"] += 1; continue
        actual = sum(float(r.get(f) or 0) for f in STAT[b["market"]])
        d = actual - b["line"]
        if abs(d) < 1e-9:
            outcome = "push"
        else:
            outcome = "win" if (d > 0) == (b["side"] == "over") else "loss"
        p_mult = no_vig_pair(b["price"], b["opp_price"])[0]
        p_pow = power_devig_pair(b["price"], b["opp_price"])[0]
        out.append({**b, "actual": actual, "outcome": outcome, "p_mult": p_mult, "p_power": p_pow,
                    "hour_central": (b["t"] + CENTRAL).hour, "t": b["t"].isoformat()})
    return out, skipped


def grade(rows, reps=2000, seed=12):
    d = [r for r in rows if r["outcome"] in ("win", "loss")]
    n = len(d); w = sum(r["outcome"] == "win" for r in d)
    out = {"bets": len(rows), "decided": n, "wins": w, "hit_rate": round(w / n, 4) if n else None, "hit_95ci": wilson(w, n)}
    if not n:
        return out
    pnl = [profit_per_unit(r["price"]) if r["outcome"] == "win" else -1.0 for r in d]
    out["roi_pct"] = round(100 * sum(pnl) / n, 2)
    out["avg_breakeven"] = round(sum(1 / (1 + profit_per_unit(r["price"])) for r in d) / n, 4)
    for name, key in (("vs_own_book_multiplicative", "p_mult"), ("vs_own_book_power", "p_power")):
        exp = sum(r[key] for r in d); var = sum(r[key] * (1 - r[key]) for r in d)
        out[name] = {"expected": round(exp / n, 4), "excess": round((w - exp) / n, 4),
                     "z": round((w - exp) / math.sqrt(var), 2) if var else None}
    if all("fair" in r for r in d):
        exp = sum(r["fair"] for r in d); var = sum(r["fair"] * (1 - r["fair"]) for r in d)
        out["vs_consensus"] = {"expected": round(exp / n, 4), "excess": round((w - exp) / n, 4),
                               "z": round((w - exp) / math.sqrt(var), 2) if var else None}
    by = defaultdict(list)
    for r, p in zip(d, pnl):
        by[r["game_id"]].append(p)
    groups = list(by.values())
    if len(groups) >= 10:
        rng = np.random.default_rng(seed)
        boots = []
        for _ in range(reps):
            pick = rng.integers(0, len(groups), len(groups))
            tot = sum(sum(groups[i]) for i in pick); cnt = sum(len(groups[i]) for i in pick)
            boots.append(100 * tot / cnt)
        out["game_cluster_roi_95ci_pct"] = [round(float(np.percentile(boots, 2.5)), 2), round(float(np.percentile(boots, 97.5)), 2)]
        out["games"] = len(groups)
    return out


def checks(g):
    P = FROZEN["pass"]
    return {"decided_ge_300": g["decided"] >= P["min_bets"],
            "roi_gt_0": (g.get("roi_pct") if g.get("roi_pct") is not None else -1) > P["min_roi_pct"],
            "z_ge_2.5_vs_own_book_multiplicative": ((g.get("vs_own_book_multiplicative") or {}).get("z") or -9) >= P["min_z"],
            "z_ge_2.5_vs_own_book_power": ((g.get("vs_own_book_power") or {}).get("z") or -9) >= P["min_z"],
            "game_cluster_ci_low_gt_0": (g.get("game_cluster_roi_95ci_pct") or [-1])[0] > P["min_cluster_ci_low_pct"]}


def run(props, stats, validated=True):
    if validated:
        require_eligible(props, "R12 book disagreement")
    groups = opening_quotes(props)
    vendors = Counter(v for books in groups.values() for v in books)
    books_per_group = Counter(min(len(b), 6) for b in groups.values())
    pb, lb, seen = pick_bets(groups)
    res = {"hypotheses": {}}
    for name, bets in (("H12a_price_outlier", pb), ("H12b_line_outlier", lb)):
        rows, skipped = settle(bets, stats)
        g = grade(rows)
        c = checks(g)
        res["hypotheses"][name] = {
            "verdict": "PASS" if all(c.values()) else "FAIL", "checks": c, "primary": g, "skipped": dict(skipped),
            "descriptive_by_market": {m: grade([r for r in rows if r["market"] == m]) for m in FROZEN["markets"]},
            "descriptive_by_side": {s: grade([r for r in rows if r["side"] == s]) for s in ("over", "under")},
            "descriptive_by_outlier_vendor": {v: grade([r for r in rows if r["vendor"] == v]) for v in sorted({r["vendor"] for r in rows})},
            "descriptive_by_posted_hour_central": {h: len([r for r in rows if r["hour_central"] == h]) for h in range(24)},
            "sample": rows[:25]}
    res.update({"frozen": FROZEN, "evidence_class": "validated" if validated else "exploratory_unverified_prices",
                "coverage": {"vendors_quoting": dict(vendors), "books_per_player_market": dict(sorted(books_per_group.items())),
                             **dict(seen)},
                "limitations": ["Opening quotes only: we do not observe whether the outlier price was still available "
                                "later in the day. The 60-minute window limits, but does not remove, that risk.",
                                "Only the live logger (simultaneous snapshots) can confirm a daily-check bettor could get these prices.",
                                "A PASS licenses only a frozen 2026-27 shadow test; it is not yet a betting rule."]})
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--props", required=True)
    p.add_argument("--stats", action="append", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    res = run(read(a.props), [s for f in a.stats for s in read(f)])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2, default=str) + "\n")
    for name, h in res["hypotheses"].items():
        g = h["primary"]
        print(name, h["verdict"], "decided", g["decided"], "hit", g.get("hit_rate"), "roi%", g.get("roi_pct"),
              "ci", g.get("game_cluster_roi_95ci_pct"))
    print(json.dumps(res["coverage"], indent=2))


if __name__ == "__main__":
    main()
