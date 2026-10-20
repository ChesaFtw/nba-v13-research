"""H1 (pre-registered in PREREG_H1_BLOWOUT_MINUTES.md): do opening prop prices under-price
blowout-driven minutes loss for high-minute starters?

Mechanism: in lopsided games starters sit late, so their minutes have a fat left tail that
depends on the pre-game spread. If books build starter prop lines mostly from per-game
averages, UNDERS on those starters in high-|spread| games should beat the de-vigged price.

Inputs must be provenance-gated (scripts/quarantine_odds.py): eligible opening props, eligible
opening game odds, played-minutes box scores (regular season, min > 0), and the game index.
Frozen constants are below; nothing is tuned by this script.
"""
import argparse, json, math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from scripts.backtest_opening_spreads_totals import choose
from scripts.backtest_prop_holdout import ALIASES, VENDORS
from scripts.regular_season_filter import minutes as parse_minutes
from v13.pricing import no_vig_pair, power_devig_pair, profit_per_unit, prop_prices, tip_times, wilson
from v13.provenance import require_eligible

FROZEN = {"starter_min_avg": 30.0, "starter_window": 10, "starter_min_games": 5,
          "blowout_abs_spread": 9.5, "competitive_abs_spread": 4.5,
          "markets": ("points", "rebounds", "assists", "threes"),
          "pass": {"min_bets": 300, "min_excess": 0.02, "min_z": 2.5, "min_roi_pct": 0.0, "min_cluster_ci_low_pct": 0.0}}
STAT = {"points": ("pts",), "rebounds": ("reb",), "assists": ("ast",), "threes": ("fg3m",)}


def read(p):
    return [json.loads(x) for x in Path(p).open() if x.strip()]


def player_history(stats):
    """player_id -> sorted list of (date, game_id, team_id, minutes, row)."""
    h = defaultdict(list)
    for r in stats:
        g, pl, tm = r.get("game") or {}, r.get("player") or {}, r.get("team") or {}
        if pl.get("id") is None or g.get("id") is None:
            continue
        h[str(pl["id"])].append((str(g.get("date"))[:10], str(g["id"]), str(tm.get("id")), parse_minutes(r.get("min")), r))
    for v in h.values():
        v.sort(key=lambda x: (x[0], x[1]))
    return h


def prop_lines(props, tips):
    """One line per (game, player, market): fixed vendor priority, earliest gated row."""
    best = {}
    for r in props:
        m = ALIASES.get(r.get("prop_type"))
        if m not in FROZEN["markets"] or (r.get("market") or {}).get("type") != "over_under":
            continue
        over, under = prop_prices(r)
        if over is None or under is None or r.get("line_value") is None:
            continue
        key = (str(r["game_id"]), str(r["player_id"]), m)
        v = str(r.get("vendor") or "").lower()
        rank = (VENDORS.index(v) if v in VENDORS else 99, str(r.get("opened_at")))
        if key not in best or rank < best[key][0]:
            best[key] = (rank, {"line": float(r["line_value"]), "over": over, "under": under})
    return {k: v[1] for k, v in best.items()}


def build(props, game_odds, stats, games):
    tips = tip_times(games)
    gl = choose(game_odds, tips)
    home = {str(g["id"]): str((g.get("home_team") or {}).get("id")) for g in games}
    hist = player_history(stats)
    rows, skipped = [], Counter()
    for (gid, pid, market), pl in prop_lines(props, tips).items():
        o = gl.get(gid)
        if not o or o.get("spread_home_value") is None:
            skipped["no_gated_game_line"] += 1; continue
        h = hist.get(pid, [])
        idx = next((i for i, x in enumerate(h) if x[1] == gid), None)
        if idx is None:
            skipped["did_not_play_or_no_box"] += 1; continue  # book voids DNP props
        prior = [x for x in h[:idx] if x[0] < h[idx][0]][-FROZEN["starter_window"]:]
        if len(prior) < FROZEN["starter_min_games"]:
            skipped["insufficient_history"] += 1; continue
        avg_min = sum(x[3] for x in prior) / len(prior)
        if avg_min < FROZEN["starter_min_avg"]:
            skipped["not_high_minute_starter"] += 1; continue
        team = prior[-1][2]  # team known before the game
        spread_home = float(o["spread_home_value"])
        team_margin = -spread_home if team == home.get(gid) else spread_home
        a = abs(team_margin)
        group = "blowout" if a >= FROZEN["blowout_abs_spread"] else ("competitive" if a <= FROZEN["competitive_abs_spread"] else "middle")
        actual = sum(float(h[idx][4].get(f) or 0) for f in STAT[market])
        d = actual - pl["line"]
        outcome = "push" if abs(d) < 1e-9 else ("win" if d < 0 else "loss")  # UNDER is the tested side
        rows.append({"game_id": gid, "player_id": pid, "market": market, "group": group, "team_margin": team_margin,
                     "line": pl["line"], "price": pl["under"], "opp_price": pl["over"], "outcome": outcome,
                     "minutes": h[idx][3], "avg_min_prior": avg_min,
                     "p_mult": no_vig_pair(pl["under"], pl["over"])[0], "p_power": power_devig_pair(pl["under"], pl["over"])[0]})
    return rows, skipped


def grade(rows, reps=2000, seed=3):
    d = [r for r in rows if r["outcome"] in ("win", "loss")]
    n = len(d); w = sum(r["outcome"] == "win" for r in d)
    out = {"bets": len(rows), "decided": n, "wins": w, "hit_95ci": wilson(w, n)}
    if not n:
        return out
    pnl = [profit_per_unit(r["price"]) if r["outcome"] == "win" else -1.0 for r in d]
    out["roi_pct"] = round(100 * sum(pnl) / n, 2)
    for name, key in (("multiplicative", "p_mult"), ("power", "p_power")):
        exp = sum(r[key] for r in d); var = sum(r[key] * (1 - r[key]) for r in d)
        out[name] = {"expected": round(exp / n, 4), "excess": round((w - exp) / n, 4),
                     "z": round((w - exp) / math.sqrt(var), 2) if var else None}
    by = defaultdict(list)
    for r, p in zip(d, pnl):
        by[r["game_id"]].append(p)
    groups = list(by.values())
    if len(groups) >= 10:
        rng = np.random.default_rng(seed)
        boots = [100 * sum(sum(g) for g in (groups[i] for i in idx)) / sum(len(g) for g in (groups[i] for i in idx))
                 for idx in (rng.integers(0, len(groups), len(groups)) for _ in range(reps))]
        out["game_cluster_roi_95ci_pct"] = [round(float(np.percentile(boots, 2.5)), 2), round(float(np.percentile(boots, 97.5)), 2)]
        out["games"] = len(groups)
    return out


def run(props, game_odds, stats, games, validated=True):
    if validated:
        require_eligible(props, "H1 props"); require_eligible(game_odds, "H1 game odds")
    rows, skipped = build(props, game_odds, stats, games)
    blow = [r for r in rows if r["group"] == "blowout"]
    comp = [r for r in rows if r["group"] == "competitive"]
    primary, control = grade(blow), grade(comp)
    mech = {g: (round(float(np.mean([r["minutes"] - r["avg_min_prior"] for r in rs])), 2) if rs else None)
            for g, rs in (("blowout", blow), ("competitive", comp))}
    P = FROZEN["pass"]
    checks = {
        "bets_ge_300": primary["decided"] >= P["min_bets"],
        "roi_gt_0": primary.get("roi_pct", -1) > P["min_roi_pct"],
        "excess_ge_2pts_multiplicative": (primary.get("multiplicative") or {}).get("excess", -1) >= P["min_excess"],
        "excess_ge_2pts_power": (primary.get("power") or {}).get("excess", -1) >= P["min_excess"],
        "z_ge_2.5_multiplicative": ((primary.get("multiplicative") or {}).get("z") or -9) >= P["min_z"],
        "z_ge_2.5_power": ((primary.get("power") or {}).get("z") or -9) >= P["min_z"],
        "game_cluster_ci_low_gt_0": (primary.get("game_cluster_roi_95ci_pct") or [-1])[0] > P["min_cluster_ci_low_pct"],
        "mechanism_minutes_lower_in_blowouts": mech["blowout"] is not None and mech["competitive"] is not None
                                               and mech["blowout"] < mech["competitive"],
        "excess_higher_than_competitive_control": ((primary.get("power") or {}).get("excess", -1)
                                                   > (control.get("power") or {}).get("excess", 1)),
    }
    return {"hypothesis": "H1_blowout_minutes_unders", "frozen": FROZEN,
            "evidence_class": "validated" if validated else "exploratory_unverified_prices",
            "verdict": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
            "primary_blowout_unders": primary, "control_competitive_unders": control,
            "minutes_minus_prior_avg": mech, "skipped": dict(skipped),
            "descriptive_by_market": {m: grade([r for r in blow if r["market"] == m]) for m in FROZEN["markets"]}}


def main():
    p = argparse.ArgumentParser()
    for k in ("props", "game-odds", "stats", "games", "out"):
        p.add_argument(f"--{k}", required=True)
    a = p.parse_args()
    res = run(read(a.props), read(a.game_odds), read(a.stats), read(a.games))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.out).write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=2)); print(f"VERDICT: {res['verdict']}")


if __name__ == "__main__":
    main()
