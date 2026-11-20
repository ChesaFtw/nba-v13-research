"""Does a pre-registered feature carry information the de-vigged opening price does not?

For each (market, feature) in REGISTRY:
  baseline   logit P(over) = a + b * logit(p_market)            (the price, recalibrated)
  challenger logit P(over) = a + b * logit(p_market) + c * z(feature)
Both fitted on TRAIN (dates < split); scored on TEST (dates >= split) by log loss.
  delta = test log loss(challenger) - test log loss(baseline)    (negative = information)
Uncertainty: game-cluster bootstrap of delta (props from one game are correlated).
Multiple testing: Benjamini-Hochberg at q = 0.10 over every (market, feature) test.
A feature PASSES only if: delta < 0, BH-significant, and the sign of c is the same when
fitted separately on the two halves of TRAIN (stability). Nothing here is a betting rule.

Inputs: provenance-gated opening props, played-minutes box scores (all seasons available, so
head-to-head history reaches back), game index(es).
"""
import argparse, json, math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from scripts.backtest_prop_holdout import ALIASES, VENDORS
from v13.matchup_features import STATS, build
from v13.pricing import power_devig_pair, prop_prices
from v13.provenance import require_eligible

REGISTRY = {
    "rebounds": ["height_diff", "opp_missed_fg", "own_missed_fg", "height_x_oppmiss", "opp_fg3a_rate"],
    "points": ["h2h_resid_points", "years_since_draft", "yrs_x_b2b"],
    "assists": ["h2h_resid_assists", "years_since_draft", "yrs_x_b2b"],
}
SPLIT_DATE = "2026-02-01"
FDR_Q = 0.10
BOOT = 1000


def read(p):
    return [json.loads(x) for x in Path(p).open() if x.strip()]


def logit(p):
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def fit_logistic(X, y, ridge=1e-3, iters=60):
    beta = np.zeros(X.shape[1])
    pen = np.eye(X.shape[1]) * ridge; pen[0, 0] = 0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ beta)))
        g = X.T @ (y - p) - pen @ beta
        H = X.T @ (X * (p * (1 - p))[:, None]) + pen
        step = np.linalg.solve(H, g); beta += step
        if np.max(np.abs(step)) < 1e-9:
            break
    return beta


def loglik_terms(X, y, beta):
    p = np.clip(1 / (1 + np.exp(-(X @ beta))), 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def candidates(props, stats, games):
    feats = build(stats, games)
    actual = {}
    for r in stats:
        g, pl = r.get("game") or {}, r.get("player") or {}
        if g.get("id") is not None and pl.get("id") is not None:
            actual[(str(g["id"]), str(pl["id"]))] = r
    date_of = {str(g["id"]): str(g["date"])[:10] for g in games}
    best = {}
    for r in props:
        m = ALIASES.get(r.get("prop_type"))
        if m not in REGISTRY or (r.get("market") or {}).get("type") != "over_under":
            continue
        over, under = prop_prices(r)
        if over is None or under is None or r.get("line_value") is None:
            continue
        key = (str(r["game_id"]), str(r["player_id"]), m)
        v = str(r.get("vendor") or "").lower()
        rank = (VENDORS.index(v) if v in VENDORS else 99, str(r.get("opened_at")))
        if key not in best or rank < best[key][0]:
            best[key] = (rank, float(r["line_value"]), over, under)
    out, skipped = [], Counter()
    for (gid, pid, m), (_, line, over, under) in best.items():
        box = actual.get((gid, pid))
        if box is None:
            skipped["no_box_score_(DNP_void)"] += 1; continue
        val = float(box.get(STATS[m]) or 0)
        if abs(val - line) < 1e-9:
            skipped["push"] += 1; continue
        f = feats.get((gid, pid))
        if f is None:
            skipped["no_features"] += 1; continue
        out.append({"game_id": gid, "date": date_of.get(gid, ""), "market": m, "y": int(val > line),
                    "lp": logit(power_devig_pair(over, under)[0]), "f": f})
    return out, skipped


def evaluate_feature(rows, feature, split=SPLIT_DATE, seed=11):
    rows = [r for r in rows if r["f"].get(feature) is not None]
    tr = [r for r in rows if r["date"] < split]; te = [r for r in rows if r["date"] >= split]
    if len(tr) < 500 or len(te) < 300:
        return {"status": "insufficient", "train": len(tr), "test": len(te)}
    mu = np.mean([r["f"][feature] for r in tr]); sd = np.std([r["f"][feature] for r in tr])
    if sd < 1e-12:
        return {"status": "no_variance", "train": len(tr), "test": len(te)}
    def mats(rs, with_f):
        cols = [np.ones(len(rs)), np.array([r["lp"] for r in rs])]
        if with_f:
            cols.append((np.array([r["f"][feature] for r in rs]) - mu) / sd)
        return np.column_stack(cols), np.array([r["y"] for r in rs], dtype=float)
    Xb, y = mats(tr, False); Xc, _ = mats(tr, True)
    bb, bc = fit_logistic(Xb, y), fit_logistic(Xc, y)
    half = sorted(tr, key=lambda r: r["date"]); h1, h2 = half[:len(half) // 2], half[len(half) // 2:]
    c1 = fit_logistic(*mats(h1, True))[2]; c2 = fit_logistic(*mats(h2, True))[2]
    Tb, yt = mats(te, False); Tc, _ = mats(te, True)
    d = loglik_terms(Tc, yt, bc) - loglik_terms(Tb, yt, bb)
    games = defaultdict(list)
    for r, di in zip(te, d):
        games[r["game_id"]].append(di)
    g = [np.array(v) for v in games.values()]
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(BOOT):
        pick = [g[i] for i in rng.integers(0, len(g), len(g))]
        boots.append(np.concatenate(pick).mean())
    boots = np.array(boots)
    return {"status": "tested", "train": len(tr), "test": len(te), "test_games": len(g),
            "coef_std": round(float(bc[2]), 4), "coef_half1": round(float(c1), 4), "coef_half2": round(float(c2), 4),
            "sign_stable": bool(np.sign(c1) == np.sign(c2) == np.sign(bc[2])),
            "delta_logloss": round(float(d.mean()), 6),
            "delta_95ci": [round(float(np.percentile(boots, 2.5)), 6), round(float(np.percentile(boots, 97.5)), 6)],
            "p_one_sided": round(float((boots >= 0).mean()), 4),
            "baseline_test_logloss": round(float(loglik_terms(Tb, yt, bb).mean()), 5)}


def benjamini_hochberg(pvals, q=FDR_Q):
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    k = 0
    for rank, i in enumerate(order, 1):
        if pvals[i] <= q * rank / len(pvals):
            k = rank
    passed = set(order[:k])
    return [i in passed for i in range(len(pvals))]


def run(props, stats, games, validated=True):
    if validated:
        require_eligible(props, "feature information test")
    rows, skipped = candidates(props, stats, games)
    results, keys, pv = {}, [], []
    for m, fs in REGISTRY.items():
        mr = [r for r in rows if r["market"] == m]
        for f in fs:
            res = evaluate_feature(mr, f)
            results[f"{m}|{f}"] = res
            if res["status"] == "tested":
                keys.append(f"{m}|{f}"); pv.append(res["p_one_sided"])
    sig = benjamini_hochberg(pv) if pv else []
    for k, s in zip(keys, sig):
        r = results[k]
        r["bh_significant"] = s
        r["verdict"] = "PASS" if (s and r["delta_logloss"] < 0 and r["sign_stable"]) else "FAIL"
    return {"registry": REGISTRY, "split_date": SPLIT_DATE, "fdr_q": FDR_Q, "tests": len(keys),
            "evidence_class": "validated" if validated else "exploratory_unverified_prices",
            "rows_by_market": dict(Counter(r["market"] for r in rows)), "skipped": dict(skipped),
            "results": results, "passing": [k for k in keys if results[k]["verdict"] == "PASS"],
            "reading_guide": "PASS = the feature improved out-of-sample log loss beyond the de-vigged opening price, "
                             "survived BH-FDR and kept its sign in both training halves. It is information, not yet a bet."}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--props", required=True)
    p.add_argument("--stats", action="append", required=True)
    p.add_argument("--games", action="append", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    res = run(read(a.props), [s for f in a.stats for s in read(f)], [g for f in a.games for g in read(f)])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.out).write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "results"}, indent=2))
    for k, v in res["results"].items():
        print(k, v.get("verdict", v["status"]), v.get("delta_logloss"), v.get("delta_95ci"), v.get("p_one_sided"))


if __name__ == "__main__":
    main()
