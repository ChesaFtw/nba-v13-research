"""Round 14 (pre-registered in PREREG_R14_THREES_REBOUNDS.md): three-pointers and rebounds.

1. R13 reproduction and diagnosis of the threes PASS (descriptive; the same window cannot confirm it).
2. Part A — prediction only, no prices, walk-forward folds (train 2023 -> test 2024; train 2023+2024 -> test 2025):
   threes: attempts x shrunk conversion (V1) vs R13 minutes x 3PM-rate (V0), by log score of actual 3PM;
   rebounds: opportunity -> share -> GLM (with height x misses x minutes interactions) vs R13 baseline,
   and the GLM with vs without the physical terms (does height carry predictive value at all?).
3. Part B — information beyond the de-vigged 2025-26 opening price (screening; BH across R14 tests).
4. Replication of the R13 threes edge on 2024-25 opening props (quarantined prices: exploratory only).
5. Vacated-minutes audit (trades, same-night returns, long absences).
6. Tracking tests run only if the frozen coverage gate passes.
"""
import argparse, json, math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from scripts.backtest_prop_holdout import ALIASES, VENDORS
from scripts.feature_info_test import BOOT, FDR_Q, benjamini_hochberg, fit_logistic, loglik_terms, logit
from v13.minutes_model import MinutesModel, nb_over_prob
from v13.minutes_model import build as build_minutes
from v13.pricing import power_devig_pair, profit_per_unit, prop_prices
from v13.provenance import require_eligible
from v13.r14_features import build as build_r14
from v13.r14_models import PoissonGLM, nb_dispersion, nb_logpmf, threes_logpmf, threes_over_prob

SPLIT_2025 = "2026-02-01"
SPLIT_2024 = "2025-02-01"
FOLDS = {"fold_2024": ((2023,), 2024), "fold_2025": ((2023, 2024), 2025)}
REB_PHYS = ("opp_size", "osz_x_oppmiss", "osz_x_height", "osz_x_min")
REB_OTHER = ("vacated_reb", "opp_oreb_rate", "opp_fg3a_rate", "b2b", "three_in_four")
TESTS = [("threes", "edge_v1", ()), ("threes", "edge_v1", ("edge_v0",)), ("threes", "attempt_trend", ()),
         ("threes", "hot_gap", ()), ("threes", "vacated_ast", ()),
         ("rebounds", "edge_struct", ()), ("rebounds", "edge_struct", ("edge_b0",)), ("rebounds", "vacated_reb", ())]
TRACKING_TESTS = [("rebounds", "edge_track", ()), ("rebounds", "edge_track", ("edge_struct",))]
GATE = {"min_coverage": 0.90, "min_consistency": 0.95, "seasons": (2023, 2024, 2025)}
MIN_TRAIN, MIN_TEST = 500, 300


def read(p):
    return [json.loads(x) for x in Path(p).open() if x.strip()]


def cluster_ci(values, groups, reps=BOOT, seed=14):
    by = defaultdict(list)
    for v, g in zip(values, groups):
        by[g].append(v)
    arr = [np.array(v) for v in by.values()]
    if len(arr) < 10:
        return None
    rng = np.random.default_rng(seed)
    boots = [np.concatenate([arr[i] for i in rng.integers(0, len(arr), len(arr))]).mean() for _ in range(reps)]
    return [round(float(np.percentile(boots, 2.5)), 5), round(float(np.percentile(boots, 97.5)), 5)]


# --------------------------------------------------------------------------- items and folds
def make_items(stats, games, tracking=None):
    m13 = build_minutes(stats, games)
    f14 = build_r14(stats, games, tracking)
    box = {}
    for r in stats:
        g, pl = r.get("game") or {}, r.get("player") or {}
        if g.get("id") is not None and pl.get("id") is not None:
            box[(str(g["id"]), str(pl["id"]))] = r
    items = []
    for (gid, pid), v in m13.items():
        r = box.get((gid, pid), {})
        f = dict(v["features"]); f.update(f14.get((gid, pid), {}))
        items.append({"gid": gid, "pid": pid, "date": v["date"], "season": v["season"], "minutes": v["minutes"],
                      "features": v["features"], "f": f, "rates": v["rates"], "stats": v["stats"],
                      "fg3a": float(r.get("fg3a") or 0)})
    return items


def _num(v, default):
    return default if v is None else float(v)


def fit_fold(train, test):
    mm = MinutesModel().fit(train)
    out = {"minutes_model": mm}
    for name, items in (("train", train), ("test", test)):
        for it, p in zip(items, mm.predict_mean(items)):
            it["_pm"] = float(p)
    # threes V0 (R13) and V1 (attempts x conversion)
    tr0 = [it for it in train if it["rates"].get("threes") is not None]
    out["phi_v0"] = nb_dispersion([it["_pm"] * it["rates"]["threes"] for it in tr0], [it["stats"]["threes"] for it in tr0])
    tr1 = [it for it in train if it["f"].get("fg3a_pm") is not None]
    out["phi_att"] = nb_dispersion([it["_pm"] * it["f"]["fg3a_pm"] for it in tr1], [it["fg3a"] for it in tr1])
    # rebounds B0 (R13) and structural GLM
    trb = [it for it in train if it["rates"].get("rebounds") is not None]
    out["phi_b0"] = nb_dispersion([it["_pm"] * it["rates"]["rebounds"] for it in trb], [it["stats"]["rebounds"] for it in trb])
    trs = [it for it in train if it["f"].get("exp_dreb_opps") is not None]
    if len(trs) >= 200:
        mean_of = lambda k: float(np.mean([it["f"][k] for it in trs if it["f"].get(k) is not None] or [0.0]))
        fill = {k: mean_of(k) for k in REB_OTHER}
        out["reb_centering"] = {"opp_size_in": mean_of("opp_size_in"), "height_in": mean_of("height_in"),
                                "exp_dreb_opps": mean_of("exp_dreb_opps"), "pm": float(np.mean([it["_pm"] for it in trs])), **fill}
        X = lambda its, cols: np.array([[reb_cov(it, out["reb_centering"])[c] for c in cols] for it in its])
        off = lambda its: np.array([reb_offset(it) for it in its])
        y = np.array([it["stats"]["rebounds"] for it in trs])
        out["glm_full"] = PoissonGLM().fit(X(trs, REB_PHYS + REB_OTHER), y, off(trs))
        out["glm_nophys"] = PoissonGLM().fit(X(trs, REB_OTHER), y, off(trs))
        out["phi_full"] = nb_dispersion(out["glm_full"].predict(X(trs, REB_PHYS + REB_OTHER), off(trs)), y)
        out["phi_nophys"] = nb_dispersion(out["glm_nophys"].predict(X(trs, REB_OTHER), off(trs)), y)
        out["_X"], out["_off"] = X, off
    trt = [it for it in train if it["f"].get("chances_pm") is not None and it["f"].get("chance_conv") is not None]
    if len(trt) >= 200:
        out["phi_track"] = nb_dispersion([it["_pm"] * it["f"]["chances_pm"] * it["f"]["chance_conv"] for it in trt],
                                         [it["stats"]["rebounds"] for it in trt])
    return out


def reb_offset(it):
    f = it["f"]
    return max(1e-3, it["_pm"] / 48.0 * (f["exp_dreb_opps"] * f["dreb_share"] + f["exp_oreb_opps"] * f["oreb_share"]))


def reb_cov(it, c):
    f = it["f"]
    # A player's own height is constant, so his rebound-share history already absorbs it; what varies
    # game to game is the size of the opponent he faces. Physical terms are built around that.
    osz = _num(f.get("opp_size_in"), c["opp_size_in"]) - c["opp_size_in"]
    hgt = _num(f.get("height_in"), c["height_in"]) - c["height_in"]
    om = f["exp_dreb_opps"] - c["exp_dreb_opps"]
    out = {"opp_size": osz, "osz_x_oppmiss": osz * om, "osz_x_height": osz * hgt, "osz_x_min": osz * (it["_pm"] - c["pm"])}
    for k in REB_OTHER:
        out[k] = _num(f.get(k), c[k])
    return out


def predictions(fold, it):
    """Per-item model means/dispersions; None where a model cannot be formed."""
    f, p = it["f"], {}
    r3 = it["rates"].get("threes")
    if r3 is not None:
        p["v0"] = (it["_pm"] * r3, fold["phi_v0"])
    if f.get("fg3a_pm") is not None:
        p["v1"] = (it["_pm"] * f["fg3a_pm"], fold["phi_att"], f["fg3_pct_shrunk"])
    rb = it["rates"].get("rebounds")
    if rb is not None:
        p["b0"] = (it["_pm"] * rb, fold["phi_b0"])
    if f.get("exp_dreb_opps") is not None and "glm_full" in fold:
        X, off = fold["_X"], fold["_off"]
        p["full"] = (float(fold["glm_full"].predict(X([it], REB_PHYS + REB_OTHER), off([it]))[0]), fold["phi_full"])
        p["nophys"] = (float(fold["glm_nophys"].predict(X([it], REB_OTHER), off([it]))[0]), fold["phi_nophys"])
    if f.get("chances_pm") is not None and f.get("chance_conv") is not None and "phi_track" in fold:
        p["track"] = (it["_pm"] * f["chances_pm"] * f["chance_conv"], fold["phi_track"])
    return p


# --------------------------------------------------------------------------- Part A
def part_a(fold, test):
    preds = [(it, predictions(fold, it)) for it in test]

    def mean_of(m):
        return m[0] * m[2] if len(m) == 3 else m[0]  # attempts x conversion for the threes compound model

    def compare(a, b, scorer, stat):
        rows = [(it, pr) for it, pr in preds if a in pr and b in pr]
        if len(rows) < 50:
            return {"n": len(rows), "status": "insufficient"}
        d = [scorer(it, pr[a], stat) - scorer(it, pr[b], stat) for it, pr in rows]
        mae_a = float(np.mean([abs(it["stats"][stat] - mean_of(pr[a])) for it, pr in rows]))
        mae_b = float(np.mean([abs(it["stats"][stat] - mean_of(pr[b])) for it, pr in rows]))
        ci = cluster_ci(d, [it["gid"] for it, _ in rows])
        return {"n": len(rows), "mean_logscore_gain": round(float(np.mean(d)), 5), "gain_95ci": ci,
                "mae": {a: round(mae_a, 4), b: round(mae_b, 4)}, "pass": bool(ci and ci[0] > 0)}

    def nb_score(it, m, stat):
        return nb_logpmf(it["stats"][stat], m[0], m[1])

    def t_score(it, m, stat):
        return threes_logpmf(it["stats"][stat], *m) if len(m) == 3 else nb_logpmf(it["stats"][stat], m[0], m[1])

    out = {"threes_v1_vs_v0": compare("v1", "v0", t_score, "threes"),
           "rebounds_full_vs_b0": compare("full", "b0", nb_score, "rebounds"),
           "rebounds_full_vs_nophys": compare("full", "nophys", nb_score, "rebounds"),
           "rebounds_track_vs_full": compare("track", "full", nb_score, "rebounds")}
    if "glm_full" in fold:
        names = ("intercept",) + REB_PHYS + REB_OTHER
        g = fold["glm_full"]
        out["rebound_glm_coefficients_per_sd"] = {n: {"coef": round(float(b), 4), "se": round(float(s), 4)}
                                                  for n, b, s in zip(names, g.beta_, g.se_)}
    return out


# --------------------------------------------------------------------------- Part B
def prop_rows(props, items, fold, markets=("threes", "rebounds")):
    look = {(it["gid"], it["pid"]): it for it in items}
    best = {}
    for r in props:
        m = ALIASES.get(r.get("prop_type"))
        if m not in markets or (r.get("market") or {}).get("type") != "over_under":
            continue
        over, under = prop_prices(r)
        if over is None or under is None or r.get("line_value") is None:
            continue
        key = (str(r["game_id"]), str(r["player_id"]), m)
        v = str(r.get("vendor") or "").lower()
        rank = (VENDORS.index(v) if v in VENDORS else 99, str(r.get("opened_at")))
        if key not in best or rank < best[key][0]:
            best[key] = (rank, float(r["line_value"]), over, under, v)
    rows, skipped = [], Counter()
    for (gid, pid, m), (_, line, over, under, vendor) in best.items():
        it = look.get((gid, pid))
        if it is None:
            skipped["no_box_score_or_history"] += 1; continue
        val = it["stats"][m]
        if abs(val - line) < 1e-9:
            skipped["push"] += 1; continue
        p_mkt = power_devig_pair(over, under)[0]
        lp = logit(p_mkt)
        pr = predictions(fold, it)
        f = {}
        if m == "threes":
            if "v0" in pr:
                f["edge_v0"] = logit(nb_over_prob(pr["v0"][0], pr["v0"][1], line)) - lp
            if "v1" in pr:
                f["edge_v1"] = logit(threes_over_prob(pr["v1"][0], pr["v1"][1], pr["v1"][2], line)) - lp
            for k in ("attempt_trend", "hot_gap", "vacated_ast"):
                f[k] = it["f"].get(k)
        else:
            if "b0" in pr:
                f["edge_b0"] = logit(nb_over_prob(pr["b0"][0], pr["b0"][1], line)) - lp
            if "full" in pr:
                f["edge_struct"] = logit(nb_over_prob(pr["full"][0], pr["full"][1], line)) - lp
            if "track" in pr:
                f["edge_track"] = logit(nb_over_prob(pr["track"][0], pr["track"][1], line)) - lp
            f["vacated_reb"] = it["f"].get("vacated_reb")
        rows.append({"game_id": gid, "player_id": pid, "date": it["date"], "market": m, "y": int(val > line), "lp": lp,
                     "line": line, "over": over, "under": under, "vendor": vendor, "pm": it["_pm"],
                     "vol": (it["_pm"] * it["f"]["fg3a_pm"]) if it["f"].get("fg3a_pm") is not None else None, "f": f})
    return rows, skipped


def evaluate_incremental(rows, feature, base=(), split=SPLIT_2025, seed=11, keep=False):
    """R11/R13 harness generalized: baseline = logit(market) + base features; challenger adds `feature`."""
    need = (feature,) + tuple(base)
    rows = [r for r in rows if all(r["f"].get(k) is not None for k in need)]
    tr = [r for r in rows if r["date"] < split]; te = [r for r in rows if r["date"] >= split]
    if len(tr) < MIN_TRAIN or len(te) < MIN_TEST:
        return {"status": "insufficient", "train": len(tr), "test": len(te)}
    stats = {}
    for k in need:
        v = np.array([r["f"][k] for r in tr], float)
        stats[k] = (float(v.mean()), float(v.std()))
    if stats[feature][1] < 1e-12:
        return {"status": "no_variance", "train": len(tr), "test": len(te)}

    def mats(rs, with_f):
        cols = [np.ones(len(rs)), np.array([r["lp"] for r in rs])]
        for k in tuple(base) + ((feature,) if with_f else ()):
            mu, sd = stats[k]
            cols.append((np.array([r["f"][k] for r in rs]) - mu) / (sd if sd > 1e-12 else 1.0))
        return np.column_stack(cols), np.array([r["y"] for r in rs], float)

    Xb, y = mats(tr, False); Xc, _ = mats(tr, True)
    bb, bc = fit_logistic(Xb, y), fit_logistic(Xc, y)
    half = sorted(tr, key=lambda r: r["date"]); h1, h2 = half[:len(half) // 2], half[len(half) // 2:]
    c1 = fit_logistic(*mats(h1, True))[-1]; c2 = fit_logistic(*mats(h2, True))[-1]
    Tb, yt = mats(te, False); Tc, _ = mats(te, True)
    lb, lc = loglik_terms(Tb, yt, bb), loglik_terms(Tc, yt, bc)
    d = lc - lb
    by = defaultdict(list)
    for r, di in zip(te, d):
        by[r["game_id"]].append(di)
    g = [np.array(v) for v in by.values()]
    rng = np.random.default_rng(seed)
    boots = np.array([np.concatenate([g[i] for i in rng.integers(0, len(g), len(g))]).mean() for _ in range(BOOT)])
    pc = 1 / (1 + np.exp(-(Tc @ bc))); pb = 1 / (1 + np.exp(-(Tb @ bb)))
    res = {"status": "tested", "train": len(tr), "test": len(te), "test_games": len(g),
           "coef_std": round(float(bc[-1]), 4), "coef_half1": round(float(c1), 4), "coef_half2": round(float(c2), 4),
           "sign_stable": bool(np.sign(c1) == np.sign(c2) == np.sign(bc[-1])),
           "delta_logloss": round(float(d.mean()), 6),
           "delta_95ci": [round(float(np.percentile(boots, 2.5)), 6), round(float(np.percentile(boots, 97.5)), 6)],
           "p_one_sided": round(float((boots >= 0).mean()), 4),
           "brier": {"baseline": round(float(np.mean((pb - yt) ** 2)), 6), "challenger": round(float(np.mean((pc - yt) ** 2)), 6)},
           "baseline_test_logloss": round(float(lb.mean()), 5)}
    if keep:
        res["_te"], res["_d"], res["_pc"], res["_pb"] = te, d, pc, pb
    return res


def run_tests(rows, tests, split=SPLIT_2025):
    results, keys, pv = {}, [], []
    for m, feat, base in tests:
        k = f"{m}|{feat}" + (f"|beyond:{'+'.join(base)}" if base else "")
        res = evaluate_incremental([r for r in rows if r["market"] == m], feat, base, split)
        results[k] = res
        if res["status"] == "tested":
            keys.append(k); pv.append(res["p_one_sided"])
    sig = benjamini_hochberg(pv, FDR_Q) if pv else []
    for k, s in zip(keys, sig):
        r = results[k]
        r["bh_significant"] = s
        r["verdict"] = "PASS" if (s and r["delta_logloss"] < 0 and r["delta_95ci"][1] < 0 and r["sign_stable"]) else "FAIL"
    return results, [k for k in keys if results[k]["verdict"] == "PASS"]


# --------------------------------------------------------------------------- R13 threes diagnosis
def diagnose_threes(rows, split=SPLIT_2025):
    res = evaluate_incremental([r for r in rows if r["market"] == "threes"], "edge_v0", (), split, keep=True)
    if res["status"] != "tested":
        return {"reproduction": res}
    te, d, pc, pb = res.pop("_te"), res.pop("_d"), res.pop("_pc"), res.pop("_pb")

    def agg(mask_fn):
        out = {}
        groups = defaultdict(list)
        for i, r in enumerate(te):
            groups[mask_fn(r, i)].append(i)
        for k, idx in sorted(groups.items(), key=lambda kv: str(kv[0])):
            dd = d[idx]
            out[str(k)] = {"n": len(idx), "delta_logloss": round(float(dd.mean()), 6),
                           "ci": cluster_ci(list(dd), [te[i]["game_id"] for i in idx])}
        return out

    edge = np.array([r["f"]["edge_v0"] for r in te])
    lb = lambda r, i: "0.5" if r["line"] <= 0.5 else ("1.5" if r["line"] <= 1.5 else ("2.5" if r["line"] <= 2.5 else "3.5+"))
    mb = lambda r, i: "lt20" if r["pm"] < 20 else ("20-30" if r["pm"] < 30 else "30+")
    vb = lambda r, i: "unknown" if r["vol"] is None else ("lt3" if r["vol"] < 3 else ("3-6" if r["vol"] < 6 else "6+"))
    deciles = np.quantile(edge, np.linspace(0, 1, 11))
    dec = []
    y = np.array([r["y"] for r in te])
    for k in range(10):
        m = (edge >= deciles[k]) & ((edge <= deciles[k + 1]) if k == 9 else (edge < deciles[k + 1]))
        if not m.any():
            continue
        idx = np.where(m)[0]
        pnl = []
        for i in idx:
            r = te[i]
            side_over = pc[i] > 1 / (1 + math.exp(-r["lp"]))
            price = r["over"] if side_over else r["under"]
            won = (r["y"] == 1) == side_over
            pnl.append(profit_per_unit(price) if won else -1.0)
        dec.append({"decile": k + 1, "edge_range": [round(float(deciles[k]), 3), round(float(deciles[k + 1]), 3)], "n": int(m.sum()),
                    "mean_p_market": round(float(np.mean([1 / (1 + math.exp(-te[i]["lp"])) for i in idx])), 4),
                    "mean_p_recalibrated_model": round(float(pc[m].mean()), 4), "over_hit_rate": round(float(y[m].mean()), 4),
                    "descriptive_roi_pct_side_toward_model_at_opening_price": round(100 * float(np.mean(pnl)), 2)})
    return {"reproduction": res,
            "by_line": agg(lb), "by_model_direction": agg(lambda r, i: "model_over" if r["f"]["edge_v0"] > 0 else "model_under"),
            "by_predicted_minutes": agg(mb), "by_projected_attempts": agg(vb), "by_month": agg(lambda r, i: r["date"][:7]),
            "by_vendor": agg(lambda r, i: r["vendor"]), "edge_deciles": dec,
            "note": "All slices are post-hoc on the window where the PASS was found; they explain, they cannot confirm. "
                    "ROI is descriptive at opening prices and assumes every row was bettable."}


# --------------------------------------------------------------------------- frozen shadow recipe
SHADOW_THRESHOLD = 0.03  # bet toward the recalibrated model only when it differs from the de-vigged price by >= 3 pts


def freeze_threes(items, rows_2025):
    """Freeze the R13 threes recipe for 2026-27 shadow tracking. Minutes model + 3PM-rate dispersion are refit
    on every completed season (2023-24..2025-26); the recalibration (logit market + standardized edge) is fit on
    all 2025-26 threes rows from the walk-forward fold. Nothing here is a validated betting rule."""
    full = [dict(it) for it in items if it["season"] in (2023, 2024, 2025)]
    if len(full) < 1000:
        return {"status": "insufficient"}
    mm = MinutesModel().fit(full)
    for it, p in zip(full, mm.predict_mean(full)):
        it["_pm"] = float(p)
    tr0 = [it for it in full if it["rates"].get("threes") is not None]
    phi = nb_dispersion([it["_pm"] * it["rates"]["threes"] for it in tr0], [it["stats"]["threes"] for it in tr0])
    th = [r for r in rows_2025 if r["market"] == "threes" and r["f"].get("edge_v0") is not None]
    if len(th) < 500:
        return {"status": "insufficient_rows", "rows": len(th)}
    e = np.array([r["f"]["edge_v0"] for r in th]); mu, sd = float(e.mean()), float(e.std())
    X = np.column_stack([np.ones(len(th)), [r["lp"] for r in th], (e - mu) / sd])
    beta = fit_logistic(X, np.array([r["y"] for r in th], float))
    return {"status": "frozen", "minutes_model_coefficients": mm.coefficients(),
            "minutes_model_standardization": {"mean": [round(float(v), 6) for v in mm.mu], "sd": [round(float(v), 6) for v in mm.sd]},
            "phi_threes": round(phi, 6), "edge_standardization": {"mean": round(mu, 6), "sd": round(sd, 6)},
            "recalibration_logit": {"intercept": round(float(beta[0]), 6), "market_logit": round(float(beta[1]), 6),
                                    "edge_z": round(float(beta[2]), 6)},
            "decision_rule": f"bet the side toward the recalibrated probability when |p_recal - p_devig| >= {SHADOW_THRESHOLD}, "
                             "at the price observed by the logger at the daily scan time; flat 1 unit; shadow only",
            "rows_used_for_recalibration": len(th), "trained_on_seasons": [2023, 2024, 2025]}


# --------------------------------------------------------------------------- vacated-minutes audit
def vacated_audit(stats, games, season=2025, rotation_min=15.0):
    """How often does the R13/R14 'absent from the previous game' proxy mislead? (diagnostic; uses
    post-game box scores on purpose, so it is never a feature)."""
    import bisect
    from scripts.regular_season_filter import minutes as pm
    sched = defaultdict(list)
    for g in games:
        if g.get("season") != season or g.get("postseason"):
            continue
        for side in ("home_team", "visitor_team"):
            t = (g.get(side) or {}).get("id")
            if t is not None:
                sched[str(t)].append((str(g["date"])[:10], str(g["id"])))
    for t in sched:
        sched[t].sort()
    appear = defaultdict(list)
    in_game = defaultdict(set)
    for r in stats:
        g = r.get("game") or {}
        pid, tid = str((r.get("player") or {}).get("id")), str((r.get("team") or {}).get("id"))
        appear[pid].append((str(g.get("date"))[:10], str(g.get("id")), tid, pm(r.get("min"))))
        in_game[str(g.get("id"))].add(pid)
    for v in appear.values():
        v.sort()
    dates = {pid: [a[0] for a in ap] for pid, ap in appear.items()}
    events = sorted((d, tid, i) for tid, gs in sched.items() for i, (d, _) in enumerate(gs) if i > 0)
    last = {}                        # pid -> index into appear[pid] of most recent appearance before the day
    roster = defaultdict(set)
    counts = Counter()
    pointer = {pid: 0 for pid in appear}
    def advance(day):
        for pid, ap in appear.items():
            j = pointer[pid]
            while j < len(ap) and ap[j][0] < day:
                old = last.get(pid)
                if old is not None:
                    roster[ap[old][2]].discard(pid)
                last[pid] = j; roster[ap[j][2]].add(pid); j += 1
            pointer[pid] = j
    current_day = None
    for day, tid, i in events:
        if day != current_day:
            advance(day); current_day = day
        gs = sched[tid]; gid, prev = gs[i][1], gs[i - 1][1]
        recent = {g_ for _, g_ in gs[max(0, i - 5):i]}
        for pid in roster[tid]:
            ap = appear[pid]; j = last[pid]
            if ap[j][1] not in recent:
                continue
            last10 = ap[max(0, j - 9):j + 1]
            if sum(a[3] for a in last10) / len(last10) < rotation_min or pid in in_game.get(prev, set()):
                continue
            counts["flagged_absences"] += 1
            if pid in in_game.get(gid, set()):
                counts["played_tonight_anyway"] += 1
            k = bisect.bisect_left(dates[pid], day)
            if k < len(ap) and ap[k][2] != tid:
                counts["next_appearance_other_team_(trade)"] += 1
            elif k >= len(ap):
                counts["never_appeared_again_in_data"] += 1
    n = counts["flagged_absences"] or 1
    return {"season": season, **dict(counts),
            "share_played_tonight_anyway": round(counts["played_tonight_anyway"] / n, 4),
            "share_trade_phantoms": round(counts["next_appearance_other_team_(trade)"] / n, 4),
            "reading": "played_tonight_anyway = the proxy said 'absent' but he played (the books knew from the injury report). "
                       "trade phantoms = teammate left via trade; the proxy keeps him on the old roster until he appears for the new team."}


# --------------------------------------------------------------------------- tracking gate
def tracking_rows(paths):
    out = {}
    for p in paths:
        for r in read(p):
            if r.get("period") not in (0, None):
                continue
            gid = (r.get("game") or {}).get("id", r.get("game_id")); pid = (r.get("player") or {}).get("id", r.get("player_id"))
            if gid is not None and pid is not None:
                out[(str(gid), str(pid))] = r
    return out


def tracking_gate(tracking, stats, games, manifests):
    season_of = {str(g["id"]): g.get("season") for g in games}
    cov = {}
    for s in GATE["seasons"]:
        keys = [(str(r["game"]["id"]), str(r["player"]["id"])) for r in stats
                if season_of.get(str(r["game"]["id"])) == s and _min(r) >= 10]
        have = sum(1 for k in keys if (tracking.get(k) or {}).get("rebound_chances_total") is not None)
        cov[str(s)] = round(have / len(keys), 4) if keys else 0.0
    cons = {str(m.get("season")): m.get("period_consistency", {}).get("rebound_chances_total") for m in manifests}
    ok = all(cov.get(str(s), 0) >= GATE["min_coverage"] for s in GATE["seasons"]) and \
        all((cons.get(str(s)) is None) or cons[str(s)] >= GATE["min_consistency"] for s in GATE["seasons"])
    return {"passed": ok, "coverage_box_player_games_min10": cov, "period_consistency": cons, "gate": GATE}


def _min(r):
    from scripts.regular_season_filter import minutes
    return minutes(r.get("min"))


# --------------------------------------------------------------------------- run
def run(stats, games, props_2025, props_2024=None, tracking=None, manifests=(), validated=True):
    if validated:
        require_eligible(props_2025, "R14 information tests")
    gate = tracking_gate(tracking, stats, games, manifests) if tracking else {"passed": False, "reason": "no tracking supplied"}
    items = make_items(stats, games, tracking if gate["passed"] else None)
    out = {"tracking_gate": gate, "part_a": {}, "evidence_class": "validated" if validated else "exploratory_unverified_prices"}
    folds = {}
    for name, (tr_s, te_s) in FOLDS.items():
        train = [dict(it) for it in items if it["season"] in tr_s]
        test = [dict(it) for it in items if it["season"] == te_s]
        if len(train) < 1000 or len(test) < 500:
            out["part_a"][name] = {"status": "insufficient", "train": len(train), "test": len(test)}
            continue
        fold = fit_fold(train, test)
        folds[name] = (fold, test)
        out["part_a"][name] = {"train": len(train), "test": len(test), **part_a(fold, test)}
    pa = [out["part_a"][k] for k in FOLDS if "threes_v1_vs_v0" in out["part_a"].get(k, {})]
    out["part_a_verdicts"] = {
        "threes_attempts_x_conversion_beats_R13": bool(pa) and len(pa) == 2 and all(x["threes_v1_vs_v0"].get("pass") for x in pa),
        "rebound_structure_beats_R13": bool(pa) and len(pa) == 2 and all(x["rebounds_full_vs_b0"].get("pass") for x in pa),
        "physical_terms_add_prediction": bool(pa) and len(pa) == 2 and all(x["rebounds_full_vs_nophys"].get("pass") for x in pa)}
    rows = []
    if "fold_2025" in folds:
        fold, test = folds["fold_2025"]
        rows, skipped = prop_rows(props_2025, test, fold)
        tests = TESTS + (TRACKING_TESTS if gate["passed"] else [])
        results, passing = run_tests(rows, tests)
        out["part_b"] = {"tests": [f"{m}|{f}" + (f"|beyond:{'+'.join(b)}" if b else "") for m, f, b in tests],
                         "rows_by_market": dict(Counter(r["market"] for r in rows)), "skipped": dict(skipped),
                         "results": results, "passing": passing, "split_date": SPLIT_2025, "fdr_q": FDR_Q}
        out["r13_threes_diagnosis"] = diagnose_threes(rows)
    if props_2024 and "fold_2024" in folds:
        fold, test = folds["fold_2024"]
        rows, skipped = prop_rows(props_2024, test, fold)
        rep = {}
        for m, feat in (("threes", "edge_v0"), ("threes", "edge_v1"), ("rebounds", "edge_b0"), ("rebounds", "edge_struct")):
            rep[f"{m}|{feat}"] = evaluate_incremental([r for r in rows if r["market"] == m], feat, (), SPLIT_2024)
        out["replication_2024_25"] = {"evidence_class": "exploratory_unverified_prices",
                                      "why": "2024-25 opening timestamps sit at tip and are outside BALLDONTLIE's documented "
                                             "opening coverage; models trained on 2023-24 only",
                                      "rows_by_market": dict(Counter(r["market"] for r in rows)), "skipped": dict(skipped),
                                      "results": rep}
    if "fold_2025" in folds:
        out["frozen_threes_shadow_recipe"] = freeze_threes(items, rows if "part_b" in out else [])
    out["vacated_audit_2025"] = vacated_audit(stats, games, 2025)
    out["cumulative_tests_on_2025_26_window"] = {"R11": 11, "R12": 2, "R13": 12,
                                                 "R14": len(out.get("part_b", {}).get("tests", []))}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stats", action="append", required=True)
    p.add_argument("--games", action="append", required=True)
    p.add_argument("--props", required=True, help="gated 2025-26 opening props")
    p.add_argument("--props-2024", help="2024-25 opening props (quarantined; exploratory replication)")
    p.add_argument("--tracking", action="append", default=[])
    p.add_argument("--tracking-manifest", action="append", default=[])
    p.add_argument("--out", required=True)
    a = p.parse_args()
    tracking = tracking_rows(a.tracking) if a.tracking else None
    manifests = [json.loads(Path(m).read_text()) for m in a.tracking_manifest]
    res = run([s for f in a.stats for s in read(f)], [g for f in a.games for g in read(f)], read(a.props),
              read(a.props_2024) if a.props_2024 else None, tracking, manifests)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2, default=str) + "\n")
    print(json.dumps({"tracking_gate": res["tracking_gate"], "part_a_verdicts": res["part_a_verdicts"]}, indent=2))
    for k, v in (res.get("part_b") or {}).get("results", {}).items():
        print(k, v.get("verdict", v["status"]), v.get("delta_logloss"), v.get("delta_95ci"), v.get("p_one_sided"))
    print("R13 reproduction:", json.dumps((res.get("r13_threes_diagnosis") or {}).get("reproduction"), default=str))


if __name__ == "__main__":
    main()
