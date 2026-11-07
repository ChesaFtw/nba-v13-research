"""Round 13 (pre-registered in PREREG_R13_OPPORTUNITY.md): minutes / opportunity model.

Part A (no prices): does a minutes model beat an exponentially weighted minutes average on unseen
games? Trained on 2023-24 + 2024-25 box scores, evaluated on all 2025-26 player-games.
Part B (prices): do the model's prop probabilities, or the opportunity features behind them, add
out-of-sample information beyond the de-vigged 2025-26 opening price? Same harness as R11
(train before SPLIT_DATE, test after, game-cluster bootstrap, BH-FDR q = 0.10, sign stability).
Neither part is a betting rule.
"""
import argparse, json, math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from scripts.backtest_prop_holdout import ALIASES, VENDORS
from scripts.feature_info_test import FDR_Q, SPLIT_DATE, benjamini_hochberg, evaluate_feature, logit
from v13.minutes_model import QUANTILES, STATS, MinutesModel, nb_dispersion, nb_over_prob
from v13.minutes_model import build as build_features
from v13.pricing import power_devig_pair, prop_prices
from v13.provenance import require_eligible

TRAIN_SEASONS = (2023, 2024)
TEST_SEASON = 2025
REGISTRY = {
    "points": ["model_edge", "vacated_min", "trend"],
    "rebounds": ["model_edge", "vacated_min"],
    "assists": ["model_edge", "vacated_min"],
    "threes": ["model_edge"],
    "pra": ["model_edge", "vacated_min", "return_flag"],
}
PASS_A = {"coverage_lo": 0.77, "coverage_hi": 0.83}
BOOT = 1000


def read(p):
    return [json.loads(x) for x in Path(p).open() if x.strip()]


def cluster_ci(values, groups, reps=BOOT, seed=13):
    by = defaultdict(list)
    for v, g in zip(values, groups):
        by[g].append(v)
    arr = [np.array(v) for v in by.values()]
    rng = np.random.default_rng(seed)
    boots = [np.concatenate([arr[i] for i in rng.integers(0, len(arr), len(arr))]).mean() for _ in range(reps)]
    return [round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)]


def part_a(train, test, model):
    y = np.array([it["minutes"] for it in test])
    pm = model.predict_mean(test)
    base = {"ewma": np.array([it["features"]["m_ewma"] for it in test]),
            "last10": np.array([it["features"]["m_l10"] for it in test])}
    q = model.predict_quantiles(test)
    gids = [it["gid"] for it in test]
    d = np.abs(y - pm) - np.abs(y - base["ewma"])
    cover = float(np.mean((y >= q[:, 0]) & (y <= q[:, 2])))
    pinball = {str(qq): round(float(np.mean(np.maximum(qq * (y - q[:, i]), (qq - 1) * (y - q[:, i])))), 4)
               for i, qq in enumerate(QUANTILES)}
    ci = cluster_ci(d, gids)
    slices = {}
    for name, mask in (("vacated_min_gt_0", [it["features"]["vacated_min"] > 0 for it in test]),
                       ("return_flag", [it["features"]["return_flag"] == 1 for it in test]),
                       ("b2b", [it["features"]["b2b"] == 1 for it in test]),
                       ("trend_abs_ge_5", [abs(it["features"]["trend"]) >= 5 for it in test])):
        m = np.array(mask)
        if m.sum():
            slices[name] = {"n": int(m.sum()), "mae_model": round(float(np.abs(y - pm)[m].mean()), 3),
                            "mae_ewma": round(float(np.abs(y - base["ewma"])[m].mean()), 3)}
    checks = {"mae_improvement_ci_upper_lt_0": ci[1] < 0, "mean_improvement_lt_0": float(d.mean()) < 0,
              "interval_80_coverage_in_range": PASS_A["coverage_lo"] <= cover <= PASS_A["coverage_hi"]}
    return {"verdict": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
            "train_player_games": len(train), "test_player_games": len(test),
            "mae": {"model": round(float(np.abs(y - pm).mean()), 3),
                    **{k: round(float(np.abs(y - v).mean()), 3) for k, v in base.items()}},
            "rmse": {"model": round(float(np.sqrt(np.mean((y - pm) ** 2))), 3),
                     **{k: round(float(np.sqrt(np.mean((y - v) ** 2))), 3) for k, v in base.items()}},
            "mae_diff_vs_ewma": round(float(d.mean()), 4), "mae_diff_95ci": ci,
            "interval_80_coverage": round(cover, 4), "pinball": pinball, "slices": slices,
            "coefficients": model.coefficients()}


def stat_projection(items, model):
    pm = model.predict_mean(items)
    out = {}
    for m in STATS:
        mu, mu_b, y = [], [], []
        for it, p in zip(items, pm):
            r = it["rates"].get(m)
            if r is None:
                continue
            mu.append(p * r); mu_b.append(it["features"]["m_ewma"] * r); y.append(it["stats"][m])
        out[m] = (np.array(mu), np.array(mu_b), np.array(y))
    return out


def candidates(props, items, model, phis):
    pm = {}
    keyed = list(items)
    for it, p in zip(keyed, model.predict_mean(keyed)):
        pm[(it["gid"], it["pid"])] = p
    look = {(it["gid"], it["pid"]): it for it in keyed}
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
    rows, skipped = [], Counter()
    for (gid, pid, m), (_, line, over, under) in best.items():
        it = look.get((gid, pid))
        if it is None:
            skipped["no_box_score_or_history"] += 1; continue
        val = it["stats"][m]
        if abs(val - line) < 1e-9:
            skipped["push"] += 1; continue
        rate = it["rates"].get(m)
        if rate is None:
            skipped["no_rate"] += 1; continue
        p_mkt = power_devig_pair(over, under)[0]
        p_mod = nb_over_prob(pm[(gid, pid)] * rate, phis[m], line)
        f = dict(it["features"]); f["model_edge"] = logit(p_mod) - logit(p_mkt)
        rows.append({"game_id": gid, "date": it["date"], "market": m, "y": int(val > line), "lp": logit(p_mkt),
                     "p_model": p_mod, "p_market": p_mkt, "f": f})
    return rows, skipped


def scores(rows):
    y = np.array([r["y"] for r in rows], float)
    out = {}
    for name, key in (("market", "p_market"), ("model", "p_model")):
        p = np.clip(np.array([r[key] for r in rows]), 1e-6, 1 - 1e-6)
        out[name] = {"log_loss": round(float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))), 5),
                     "brier": round(float(np.mean((p - y) ** 2)), 5)}
    return out


def run(stats, games, props, validated=True):
    if validated:
        require_eligible(props, "R13 opportunity information test")
    feats = build_features(stats, games)
    items = [{"gid": g, "pid": p, **v} for (g, p), v in feats.items()]
    train = [it for it in items if it["season"] in TRAIN_SEASONS]
    test = [it for it in items if it["season"] == TEST_SEASON]
    if len(train) < 1000 or len(test) < 500:
        return {"status": "insufficient", "train": len(train), "test": len(test)}
    model = MinutesModel().fit(train)
    a = part_a(train, test, model)
    proj = {}
    phis = {}
    tr = stat_projection(train, model)
    for m, (mu, mu_b, y) in tr.items():
        phis[m] = nb_dispersion(mu, y)
    for m, (mu, mu_b, y) in stat_projection(test, model).items():
        proj[m] = {"n": int(len(y)), "mae_model_minutes_x_rate": round(float(np.abs(y - mu).mean()), 3),
                   "mae_ewma_minutes_x_rate": round(float(np.abs(y - mu_b).mean()), 3), "nb_phi_train": round(phis[m], 4)}
    rows, skipped = candidates(props, test, model, phis)
    results, keys, pv = {}, [], []
    for m, fs in REGISTRY.items():
        mr = [r for r in rows if r["market"] == m]
        for f in fs:
            res = evaluate_feature(mr, f)
            results[f"{m}|{f}"] = res
            if res["status"] == "tested":
                keys.append(f"{m}|{f}"); pv.append(res["p_one_sided"])
    sig = benjamini_hochberg(pv, FDR_Q) if pv else []
    for k, s in zip(keys, sig):
        r = results[k]
        r["bh_significant"] = s
        r["verdict"] = "PASS" if (s and r["delta_logloss"] < 0 and r["delta_95ci"][1] < 0 and r["sign_stable"]) else "FAIL"
    test_rows = [r for r in rows if r["date"] >= SPLIT_DATE]
    return {"part_a_minutes": a, "stat_projection_2025": proj,
            "part_b_information": {"registry": REGISTRY, "split_date": SPLIT_DATE, "fdr_q": FDR_Q, "tests": len(keys),
                                   "rows_by_market": dict(Counter(r["market"] for r in rows)), "skipped": dict(skipped),
                                   "results": results, "passing": [k for k in keys if results[k]["verdict"] == "PASS"],
                                   "probability_scores_test_window": {m: scores([r for r in test_rows if r["market"] == m])
                                                                      for m in REGISTRY if any(r["market"] == m for r in test_rows)}},
            "evidence_class": "validated" if validated else "exploratory_unverified_prices",
            "cumulative_market_tests_on_2025_26_window": {"R11": 11, "R12": 2, "R13": len(keys),
                                                          "note": "The same 2025-26 prices have now been screened repeatedly. "
                                                                  "Anything passing is a candidate for 2026-27 prospective confirmation only."},
            "reading_guide": "Part A PASS = better minutes forecasts than a weighted average on unseen games (a prediction "
                             "result, not an edge). Part B PASS = information beyond the opening price on the test window."}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stats", action="append", required=True)
    p.add_argument("--games", action="append", required=True)
    p.add_argument("--props", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    res = run([s for f in a.stats for s in read(f)], [g for f in a.games for g in read(f)], read(a.props))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps({k: v for k, v in res.get("part_a_minutes", res).items() if k != "coefficients"}, indent=2))
    print(json.dumps(res.get("stat_projection_2025"), indent=2))
    for k, v in (res.get("part_b_information") or {}).get("results", {}).items():
        print(k, v.get("verdict", v["status"]), v.get("delta_logloss"), v.get("delta_95ci"), v.get("p_one_sided"))


if __name__ == "__main__":
    main()
