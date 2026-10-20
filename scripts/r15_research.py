"""Round 15 (pre-registered in PREREG_R15_MARKETS.md): broad market discovery on a shared opportunity foundation.

Markets: points, assists, steals, blocks, and the combos points+rebounds, points+assists, rebounds+assists.
Three-pointers are frozen (R13/R14) and are NOT retested here; their confirmation is prospective only.

0. Market feasibility: what the price files actually contain (prop types, two-sided rows, vendors).
1. Absence v2: choose k (consecutive games missed) on 2023-24 ONLY, by a frozen reliability rule.
2. Tracking field audit: non-null share by field x season x minutes role x month.
3. Part A (prediction, no prices, walk-forward folds 2023->2024 and 2023+24->2025):
   opportunity x conversion GLM vs the R13-style base (minutes x per-minute rate) for points/assists/steals/blocks;
   assists: passes-based opportunity vs recent-assists offset with the same covariates (the key hypothesis);
   combos: joint (sum) distribution vs independent convolution of the components.
4. Part B (information beyond the de-vigged 2025-26 opening price; BH across all R15 tests), reported for
   model-leans-over and model-leans-under rows separately, with calibration tables.
5. Replication of the edge tests on 2024-25 opening props (quarantined prices: exploratory only).
"""
import argparse, bisect, json, math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from scripts.backtest_prop_holdout import VENDORS
from scripts.feature_info_test import FDR_Q, benjamini_hochberg, logit
from scripts.r14_research import cluster_ci, evaluate_incremental, tracking_rows
from scripts.regular_season_filter import minutes as parse_minutes
from v13.minutes_model import MinutesModel, nb_over_prob
from v13.minutes_model import build as build_minutes
from v13.pricing import power_devig_pair, prop_prices
from v13.provenance import require_eligible
from v13.r14_models import PoissonGLM, nb_dispersion, nb_logpmf, nb_pmf_vector
from v13.r15_features import KS
from v13.r15_features import build as build_r15

SPLIT_2025, SPLIT_2024 = "2026-02-01", "2025-02-01"
FOLDS = {"fold_2024": ((2023,), 2024), "fold_2025": ((2023, 2024), 2025)}
PROP_TYPES = {"points": "points", "assists": "assists", "steals": "steals", "blocks": "blocks",
              "points_rebounds": "pr", "pts_rebs": "pr", "points_assists": "pa", "pts_asts": "pa",
              "rebounds_assists": "ra", "rebs_asts": "ra", "points_rebounds_assists": "pra", "pts_rebs_asts": "pra",
              "rebounds": "rebounds", "threes": "threes", "threes_made": "threes"}
STAT = {"points": ("pts",), "assists": ("ast",), "steals": ("stl",), "blocks": ("blk",), "rebounds": ("reb",),
        "pr": ("pts", "reb"), "pa": ("pts", "ast"), "ra": ("reb", "ast")}
BASE_RATE = {"points": "pts_pm", "assists": "ast_pm", "steals": "stl_pm", "blocks": "blk_pm", "rebounds": "reb_pm"}
COMBOS = {"pr": ("points", "rebounds"), "pa": ("points", "assists"), "ra": ("rebounds", "assists")}
TESTED_MARKETS = ("points", "assists", "steals", "blocks", "pr", "pa", "ra")
ABSENCE_RULE = {"season": 2023, "max_played_anyway": 0.25}  # every absence ends with one unknowable return game
TRACK_FIELDS = ("passes", "touches", "deflections", "defended_at_rim_fga", "usage_percentage")
FIELD_GATE = 0.90
MIN_TWO_SIDED = 2000


def read(p):
    return [json.loads(x) for x in Path(p).open() if x.strip()]


# --------------------------------------------------------------------------- 0. feasibility
UNPRICED = {"turnovers": "no turnover props in the data source", "free_throws_made": "not offered by the source",
            "field_goals_made": "not offered by the source", "three_point_attempts": "not offered by the source",
            "first_half": "not offered by the source", "alternate_lines": "milestone markets are one-sided; cannot de-vig"}


def feasibility(props):
    by = defaultdict(Counter)
    for r in props:
        t = str(r.get("prop_type")); m = r.get("market") or {}
        by[t]["rows"] += 1; by[t][f"type_{m.get('type')}"] += 1
        if m.get("type") == "over_under" and m.get("over_odds") is not None and m.get("under_odds") is not None:
            by[t]["two_sided"] += 1
            if str(r.get("vendor") or "").lower() == "caesars":
                by[t]["two_sided_caesars"] += 1
    verdict = {}
    for t, c in by.items():
        if t.endswith("_1q") or "first3min" in t:
            v = "outcome needs play-by-play (not in box scores): not modeled"
        elif t in ("double_double", "triple_double"):
            v = "milestone (one-sided) odds cannot be de-vigged: not modeled"
        elif PROP_TYPES.get(t) in TESTED_MARKETS and c["two_sided"] >= MIN_TWO_SIDED:
            v = "modeled in R15"
        elif PROP_TYPES.get(t) in ("threes",):
            v = "frozen (prospective only)"
        elif PROP_TYPES.get(t) in ("rebounds",):
            v = "R14 failed; used only as a combo component"
        elif PROP_TYPES.get(t) == "pra":
            v = "R13 model_edge failed; not retested"
        else:
            v = f"insufficient two-sided rows (< {MIN_TWO_SIDED}) or unmapped"
        verdict[t] = {**dict(c), "verdict": v}
    return {"by_prop_type": verdict, "not_available_from_source": UNPRICED}


# --------------------------------------------------------------------------- 1. absence v2 reliability
def absence_reliability(stats, games, season, k, rotation_min=15.0):
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
    idx = {t: {g_: i for i, (_, g_) in enumerate(v)} for t, v in sched.items()}
    appear = defaultdict(list); in_game = defaultdict(set)
    for r in stats:
        g = r.get("game") or {}
        pid, tid = str((r.get("player") or {}).get("id")), str((r.get("team") or {}).get("id"))
        appear[pid].append((str(g.get("date"))[:10], str(g.get("id")), tid, parse_minutes(r.get("min"))))
        in_game[str(g.get("id"))].add(pid)
    for v in appear.values():
        v.sort()
    dates = {pid: [a[0] for a in ap] for pid, ap in appear.items()}
    flagged = played = 0
    for tid, gs in sched.items():
        for i in range(1, len(gs)):
            day, gid = gs[i]
            for pid, ap in appear.items():
                j = bisect.bisect_left(dates[pid], day) - 1
                if j < 0 or ap[j][2] != tid or ap[j][1] not in idx[tid]:
                    continue
                missed = i - 1 - idx[tid][ap[j][1]]
                last10 = ap[max(0, j - 9):j + 1]
                if not (k <= missed <= 15) or sum(a[3] for a in last10) / len(last10) < rotation_min:
                    continue
                flagged += 1; played += pid in in_game.get(gid, set())
    return {"k": k, "flagged": flagged, "played_anyway": played, "share_played_anyway": round(played / flagged, 4) if flagged else None}


def choose_k(stats, games):
    rel = [absence_reliability(stats, games, ABSENCE_RULE["season"], k) for k in KS]
    ok = [r["k"] for r in rel if r["share_played_anyway"] is not None and r["share_played_anyway"] <= ABSENCE_RULE["max_played_anyway"]]
    return (min(ok) if ok else None), rel


# --------------------------------------------------------------------------- 2. tracking field audit
def tracking_field_audit(tracking, stats, games):
    season_of = {str(g["id"]): g.get("season") for g in games}
    cells = defaultdict(lambda: [0, Counter()])
    for r in stats:
        m = parse_minutes(r.get("min"))
        if m < 10:
            continue
        gid, pid = str(r["game"]["id"]), str(r["player"]["id"])
        s = season_of.get(gid); role = "starter_28+" if m >= 28 else "rotation_15-28" if m >= 15 else "bench_10-15"
        month = str(r["game"].get("date", ""))[:7]
        t = (tracking or {}).get((gid, pid)) or {}
        for key in ((s, "all", "all"), (s, role, "all"), (s, "all", month)):
            cells[key][0] += 1
            for f in TRACK_FIELDS:
                cells[key][1][f] += t.get(f) is not None
    out = defaultdict(dict)
    for (s, role, month), (n, c) in sorted(cells.items(), key=lambda kv: str(kv[0])):
        out[str(s)][f"{role}|{month}"] = {"n": n, **{f: round(c[f] / n, 4) for f in TRACK_FIELDS}}
    gate = {f: all((out.get(str(s), {}).get("all|all", {}).get(f) or 0) >= FIELD_GATE for s in (2023, 2024, 2025))
            for f in TRACK_FIELDS}
    return dict(out), gate


# --------------------------------------------------------------------------- items, models
def make_items(stats, games, tracking):
    m13 = build_minutes(stats, games)
    f15 = build_r15(stats, games, tracking)
    box = {}
    for r in stats:
        g, pl = r.get("game") or {}, r.get("player") or {}
        if g.get("id") is not None and pl.get("id") is not None:
            box[(str(g["id"]), str(pl["id"]))] = r
    items = []
    for (gid, pid), v in m13.items():
        r = box.get((gid, pid), {})
        st = {k: float(r.get(k) or 0) for k in ("pts", "reb", "ast", "stl", "blk")}
        y = {m: sum(st[k] for k in ks) for m, ks in STAT.items()}
        items.append({"gid": gid, "pid": pid, "date": v["date"], "season": v["season"], "minutes": v["minutes"],
                      "features": v["features"], "f": {**v["features"], **f15.get((gid, pid), {})}, "y": y})
    return items


COVS = {"points": ("pace_ratio", "opp_def_ratio", "vac_usage", "returning_usage", "usage_trend", "b2b", "three_in_four"),
        "assists": ("teammate_efg", "opp_ast_allowed", "pace_ratio", "vac_ast", "returning_ast", "b2b"),
        "steals": ("opp_tov_rate", "pace_ratio", "pf_per36", "b2b"),
        "blocks": ("opp_2pa_share", "pace_ratio", "pf_per36", "b2b")}


def opp_offset(m, it, use_box=False):
    f, pm = it["f"], it["_pm"]
    try:
        if m == "points":
            return pm * (f["fga2_pm"] * 2 * f["fg2_pct"] + f["fg3a_pm"] * 3 * f["fg3_pct"] + f["fta_pm"] * f["ft_pct"])
        if m == "assists":
            return pm * f["ast_pm"] if use_box else pm * f["passes_pm"] * f["ast_per_pass"]
        if m == "steals":
            return pm * f["deflections_pm"] * f["stl_per_defl"]
        if m == "blocks":
            return pm * f["rim_contests_pm"] * f["blk_per_contest"]
    except (KeyError, TypeError):
        return None
    return None


def cov_names(m, k):
    return tuple(f"{c}_{k}" if c in ("vac_usage", "vac_ast") else c for c in COVS[m]
                 if not (c in ("vac_usage", "vac_ast") and k is None))


def fit_fold(train, k, field_gate):
    mm = MinutesModel().fit(train)
    for it, p in zip(train, mm.predict_mean(train)):
        it["_pm"] = float(p)
    fold = {"mm": mm, "k": k, "phi": {}, "glm": {}, "fill": {}}
    for m, rk in BASE_RATE.items():
        tr = [it for it in train if it["f"].get(rk) is not None]
        fold["phi"][f"base_{m}"] = nb_dispersion([it["_pm"] * it["f"][rk] for it in tr], [it["y"][m] for it in tr])
    for c, (a, b) in COMBOS.items():
        tr = [it for it in train if it["f"].get(BASE_RATE[a]) is not None and it["f"].get(BASE_RATE[b]) is not None]
        fold["phi"][f"joint_{c}"] = nb_dispersion([it["_pm"] * (it["f"][BASE_RATE[a]] + it["f"][BASE_RATE[b]]) for it in tr],
                                                  [it["y"][c] for it in tr])
    needs = {"assists": ("passes",), "steals": ("deflections",), "blocks": ("defended_at_rim_fga",), "points": ()}
    for m in COVS:
        for variant in (("opp", False), ("boxoff", True)) if m == "assists" else (("opp", False),):
            name, use_box = variant
            if not use_box and not all(field_gate.get(fl, False) for fl in needs[m]):
                fold["glm"][f"{name}_{m}"] = "untestable: tracking field gate failed"
                continue
            cols = cov_names(m, k)
            tr = [it for it in train if opp_offset(m, it, use_box) is not None]
            if len(tr) < 500:
                fold["glm"][f"{name}_{m}"] = f"insufficient training rows ({len(tr)})"
                continue
            fill = {c: float(np.mean([it["f"][c] for it in tr if it["f"].get(c) is not None] or [0.0])) for c in cols}
            X = np.array([[it["f"].get(c) if it["f"].get(c) is not None else fill[c] for c in cols] for it in tr], float)
            y = np.array([it["y"][m] for it in tr]); off = np.array([max(1e-3, opp_offset(m, it, use_box)) for it in tr])
            g = PoissonGLM().fit(X, y, off)
            fold["glm"][f"{name}_{m}"] = {"model": g, "cols": cols, "fill": fill, "use_box": use_box,
                                          "phi": nb_dispersion(g.predict(X, off), y)}
    return fold


def predict(fold, it, m, kind):
    """(mean, phi) or None. kind: base | opp | boxoff | joint | indep."""
    f = it["f"]
    if kind == "base":
        rk = BASE_RATE.get(m)
        return None if rk is None or f.get(rk) is None else (it["_pm"] * f[rk], fold["phi"][f"base_{m}"])
    if kind in ("opp", "boxoff"):
        spec = fold["glm"].get(f"{kind}_{m}")
        if not isinstance(spec, dict):
            return None
        off = opp_offset(m, it, spec["use_box"])
        if off is None:
            return None
        x = np.array([[f.get(c) if f.get(c) is not None else spec["fill"][c] for c in spec["cols"]]], float)
        return float(spec["model"].predict(x, np.array([max(1e-3, off)]))[0]), spec["phi"]
    if kind == "joint":
        if m not in COMBOS:
            return None
        a, b = COMBOS[m]
        if f.get(BASE_RATE[a]) is None or f.get(BASE_RATE[b]) is None:
            return None
        return it["_pm"] * (f[BASE_RATE[a]] + f[BASE_RATE[b]]), fold["phi"][f"joint_{m}"]
    return None


def indep_logpmf(fold, it, m, k):
    a, b = COMBOS[m]
    pa, pb = predict(fold, it, a, "base"), predict(fold, it, b, "base")
    if pa is None or pb is None:
        return None
    k = int(round(k)); va = nb_pmf_vector(pa[0], pa[1], k); vb = nb_pmf_vector(pb[0], pb[1], k)
    s = float(sum(va[i] * vb[k - i] for i in range(k + 1)))
    return math.log(max(s, 1e-300))


# --------------------------------------------------------------------------- 3. Part A
def part_a(fold, test):
    mm = fold["mm"]
    for it, p in zip(test, mm.predict_mean(test)):
        it["_pm"] = float(p)
    out = {}

    def cmp(m, a, b, scorer):
        rows = []
        for it in test:
            sa, sb = scorer(it, m, a), scorer(it, m, b)
            if sa is not None and sb is not None:
                rows.append((it, sa - sb))
        if len(rows) < 100:
            return {"n": len(rows), "status": "insufficient"}
        d = [x[1] for x in rows]
        ci = cluster_ci(d, [x[0]["gid"] for x in rows])
        return {"n": len(rows), "mean_logscore_gain": round(float(np.mean(d)), 5), "gain_95ci": ci, "pass": bool(ci and ci[0] > 0)}

    def nbscore(it, m, kind):
        p = predict(fold, it, m, kind)
        return None if p is None else nb_logpmf(it["y"][m], p[0], p[1])

    def combo_score(it, m, kind):
        if kind == "joint":
            return nbscore(it, m, "joint")
        return indep_logpmf(fold, it, m, it["y"][m])

    for m in COVS:
        out[f"{m}_opp_vs_base"] = cmp(m, "opp", "base", nbscore)
    out["assists_passes_vs_recent_assists_offset"] = cmp("assists", "opp", "boxoff", nbscore)
    for c in COMBOS:
        out[f"{c}_joint_vs_independent"] = cmp(c, "joint", "indep", combo_score)
    out["glm_coefficients"] = {}
    for name, spec in fold["glm"].items():
        if isinstance(spec, dict):
            g = spec["model"]
            out["glm_coefficients"][name] = {c: {"coef_per_sd": round(float(b), 4), "se": round(float(s), 4)}
                                             for c, b, s in zip(("intercept",) + spec["cols"], g.beta_, g.se_)}
        else:
            out["glm_coefficients"][name] = spec
    return out


# --------------------------------------------------------------------------- 4. Part B
def prop_rows(props, test, fold):
    for it, p in zip(test, fold["mm"].predict_mean(test)):
        it["_pm"] = float(p)
    look = {(it["gid"], it["pid"]): it for it in test}
    best = {}
    for r in props:
        m = PROP_TYPES.get(str(r.get("prop_type")))
        if m not in TESTED_MARKETS or (r.get("market") or {}).get("type") != "over_under":
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
    k = fold["k"]
    for (gid, pid, m), (_, line, over, under, vendor) in best.items():
        it = look.get((gid, pid))
        if it is None:
            skipped["no_box_score_or_history"] += 1; continue
        val = it["y"][m]
        if abs(val - line) < 1e-9:
            skipped["push"] += 1; continue
        lp = logit(power_devig_pair(over, under)[0])
        f = {}
        for kind, name in (("base", "edge_base"), ("opp", "edge_opp"), ("joint", "edge_combo")):
            p = predict(fold, it, m, kind)
            if p is not None:
                f[name] = logit(nb_over_prob(p[0], p[1], line)) - lp
        if k is not None:
            f["vac_usage"] = it["f"].get(f"vac_usage_{k}"); f["vac_ast"] = it["f"].get(f"vac_ast_{k}")
        f["returning_usage"] = it["f"].get("returning_usage"); f["returning_ast"] = it["f"].get("returning_ast")
        rows.append({"game_id": gid, "date": it["date"], "market": m, "y": int(val > line), "lp": lp, "line": line,
                     "vendor": vendor, "f": f})
    return rows, skipped


def tests_for(k):
    t = [("points", "edge_opp", ()), ("points", "edge_opp", ("edge_base",)), ("points", "returning_usage", ()),
         ("assists", "edge_opp", ()), ("assists", "edge_opp", ("edge_base",)), ("assists", "returning_ast", ()),
         ("steals", "edge_base", ()), ("steals", "edge_opp", ()), ("steals", "edge_opp", ("edge_base",)),
         ("blocks", "edge_base", ()), ("blocks", "edge_opp", ()), ("blocks", "edge_opp", ("edge_base",)),
         ("pr", "edge_combo", ()), ("pa", "edge_combo", ()), ("ra", "edge_combo", ())]
    if k is not None:
        t += [("points", "vac_usage", ()), ("assists", "vac_ast", ())]
    return t


def calibration(pc, y, bins=10):
    q = np.quantile(pc, np.linspace(0, 1, bins + 1)); out = []
    for i in range(bins):
        m = (pc >= q[i]) & ((pc <= q[i + 1]) if i == bins - 1 else (pc < q[i + 1]))
        if m.any():
            out.append({"bin": i + 1, "n": int(m.sum()), "mean_pred": round(float(pc[m].mean()), 4), "hit": round(float(y[m].mean()), 4)})
    ece = sum(b["n"] * abs(b["mean_pred"] - b["hit"]) for b in out) / max(1, sum(b["n"] for b in out))
    return {"bins": out, "ece": round(float(ece), 4)}


def run_tests(rows, tests, split):
    results, keys, pv = {}, [], []
    for m, feat, base in tests:
        key = f"{m}|{feat}" + (f"|beyond:{'+'.join(base)}" if base else "")
        res = evaluate_incremental([r for r in rows if r["market"] == m], feat, base, split, keep=True)
        if res["status"] == "tested":
            te, d, pc, pb = res.pop("_te"), res.pop("_d"), res.pop("_pc"), res.pop("_pb")
            yv = np.array([r["y"] for r in te])
            lean_over = pc > pb
            res["by_model_direction"] = {
                name: ({"n": int(mask.sum()), "delta_logloss": round(float(d[mask].mean()), 6),
                        "ci": cluster_ci(list(d[mask]), [te[i]["game_id"] for i in np.where(mask)[0]])} if mask.any() else {"n": 0})
                for name, mask in (("leans_over", lean_over), ("leans_under", ~lean_over))}
            res["calibration_challenger"] = calibration(pc, yv)
            keys.append(key); pv.append(res["p_one_sided"])
        results[key] = res
    sig = benjamini_hochberg(pv, FDR_Q) if pv else []
    for key, s in zip(keys, sig):
        r = results[key]
        r["bh_significant"] = s
        r["verdict"] = "PASS" if (s and r["delta_logloss"] < 0 and r["delta_95ci"][1] < 0 and r["sign_stable"]) else "FAIL"
    return results, [k for k in keys if results[k]["verdict"] == "PASS"]


# --------------------------------------------------------------------------- run
def run(stats, games, props_2025, props_2024=None, tracking=None, validated=True):
    if validated:
        require_eligible(props_2025, "R15 information tests")
    out = {"feasibility_2025": feasibility(props_2025),
           "evidence_class": "validated" if validated else "exploratory_unverified_prices"}
    k, rel = choose_k(stats, games)
    out["absence_v2"] = {"rule": ABSENCE_RULE, "reliability_2023_24": rel, "chosen_k": k,
                         "note": "k chosen on 2023-24 box scores only (no prices); None = absence features dropped"}
    audit, gate = tracking_field_audit(tracking, stats, games) if tracking else ({}, {f: False for f in TRACK_FIELDS})
    out["tracking_field_audit"] = audit; out["tracking_field_gate"] = gate
    items = make_items(stats, games, tracking)
    folds = {}
    out["part_a"] = {}
    for name, (tr_s, te_s) in FOLDS.items():
        train = [dict(it) for it in items if it["season"] in tr_s]
        test = [dict(it) for it in items if it["season"] == te_s]
        if len(train) < 1000 or len(test) < 500:
            out["part_a"][name] = {"status": "insufficient"}; continue
        fold = fit_fold(train, k, gate)
        folds[name] = (fold, test)
        out["part_a"][name] = part_a(fold, test)
    out["part_a_verdicts"] = {key: all(out["part_a"].get(f, {}).get(key, {}).get("pass") for f in FOLDS)
                              for key in next((v for v in out["part_a"].values() if "points_opp_vs_base" in v), {})
                              if key != "glm_coefficients"}
    if "fold_2025" in folds:
        fold, test = folds["fold_2025"]
        rows, skipped = prop_rows(props_2025, test, fold)
        tests = tests_for(k)
        results, passing = run_tests(rows, tests, SPLIT_2025)
        out["part_b"] = {"tests": [f"{m}|{f}" + (f"|beyond:{'+'.join(b)}" if b else "") for m, f, b in tests],
                         "rows_by_market": dict(Counter(r["market"] for r in rows)), "skipped": dict(skipped),
                         "results": results, "passing": passing, "split_date": SPLIT_2025, "fdr_q": FDR_Q}
    if props_2024 and "fold_2024" in folds:
        fold, test = folds["fold_2024"]
        rows, skipped = prop_rows(props_2024, test, fold)
        rep = {}
        for m, feat, base in tests_for(k):
            if feat.startswith("edge") and not base:
                rep[f"{m}|{feat}"] = {kk: v for kk, v in evaluate_incremental([r for r in rows if r["market"] == m], feat, (), SPLIT_2024).items()
                                      if not kk.startswith("_")}
        out["replication_2024_25"] = {"evidence_class": "exploratory_unverified_prices", "rows_by_market": dict(Counter(r["market"] for r in rows)),
                                      "results": rep, "feasibility_2024": feasibility(props_2024)}
    out["promotion_rule"] = ("A Part B PASS is promoted to prospective shadow tracking only if the 2024-25 replication of the "
                             "same edge has delta_logloss < 0 and any tracking field it uses passed the field gate.")
    out["cumulative_tests_on_2025_26_window"] = {"R11": 11, "R12": 2, "R13": 12, "R14": 8, "R15": len(out.get("part_b", {}).get("tests", []))}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stats", action="append", required=True)
    p.add_argument("--games", action="append", required=True)
    p.add_argument("--props", required=True)
    p.add_argument("--props-2024")
    p.add_argument("--tracking", action="append", default=[])
    p.add_argument("--out", required=True)
    a = p.parse_args()
    res = run([s for f in a.stats for s in read(f)], [g for f in a.games for g in read(f)], read(a.props),
              read(a.props_2024) if a.props_2024 else None, tracking_rows(a.tracking) if a.tracking else None)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2, default=str) + "\n")
    print(json.dumps({"absence_v2": res["absence_v2"], "tracking_field_gate": res["tracking_field_gate"],
                      "part_a_verdicts": res["part_a_verdicts"]}, indent=2))
    for key, v in (res.get("part_b") or {}).get("results", {}).items():
        print(key, v.get("verdict", v["status"]), v.get("delta_logloss"), v.get("delta_95ci"))


if __name__ == "__main__":
    main()
