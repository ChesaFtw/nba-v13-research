"""Prospective prediction ledger (2026-27): every eligible prop, not just the bets.

Commands (all append-only; each writes new timestamped files, nothing is edited in place):
  predict  at a scan time: current BALLDONTLIE props for upcoming games (policy vendor, e.g. caesars), the injury
           snapshot, and the FROZEN recipe (R14 frozen_threes_shadow_recipe) -> one row per eligible prop with model
           probability, de-vigged market probability, EV at the offered price, projected minutes, injury context,
           qualification decision and stake (0 in shadow mode). Also writes the day's short list (markdown).
  close    shortly before tip: current quotes for predicted props (closing-line observations).
  settle   after games: actual result, outcome of the indicated side, P/L of any stake, closing-line value.
  report   running totals: CLV with game-cluster CI, log loss model vs market, ROI on staked bets.

Nothing here makes a model valid. The ledger exists so that validity can be judged on prices we really saw.
"""
import argparse, hashlib, json, math, os, sys, time
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from v13.minutes_model import FEATURES, MinutesModel, nb_over_prob
from v13.minutes_model import build as build_minutes
from v13.pricing import parse_utc, power_devig_pair, profit_per_unit, prop_prices

API = "https://api.balldontlie.io"
LEDGER_VERSION = "prop-ledger-1"
BLOCKING_STATUS = ("out", "doubtful")


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def logit(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def sigmoid(z):
    return 1 / (1 + math.exp(-z))


def read(p):
    return [json.loads(x) for x in Path(p).open() if x.strip()]


# --------------------------------------------------------------------------- frozen recipe
def minutes_model_from_recipe(recipe):
    mm = MinutesModel()
    coefs = recipe["minutes_model_coefficients"]
    mm.beta = np.array([coefs["intercept"]] + [coefs[k] for k in FEATURES], float)
    mm.mu = np.array(recipe["minutes_model_standardization"]["mean"], float)
    mm.sd = np.array(recipe["minutes_model_standardization"]["sd"], float)
    return mm


def threes_probability(recipe, pm, rate, line, p_devig_over):
    """Frozen R13/R14 threes recipe -> (p_model_raw, p_recalibrated) for the OVER."""
    p_model = nb_over_prob(pm * rate, recipe["phi_threes"], line)
    lp = logit(p_devig_over)
    edge = logit(p_model) - lp
    z = (edge - recipe["edge_standardization"]["mean"]) / recipe["edge_standardization"]["sd"]
    rc = recipe["recalibration_logit"]
    return p_model, sigmoid(rc["intercept"] + rc["market_logit"] * lp + rc["edge_z"] * z)


# --------------------------------------------------------------------------- predict (pure)
def latest_team(stats):
    last = {}
    for r in stats:
        pid, tid = (r.get("player") or {}).get("id"), (r.get("team") or {}).get("id")
        d = str((r.get("game") or {}).get("date"))[:10]
        if pid is None or tid is None:
            continue
        if str(pid) not in last or d >= last[str(pid)][0]:
            last[str(pid)] = (d, str(tid))
    return {k: v[1] for k, v in last.items()}


def build_predictions(stats, games, props, injuries, recipe, policy, captured_at, recipe_sha):
    """stats/games: all history including the current season to date. props: current prop rows for upcoming games."""
    vendor = policy["vendor"]
    tips = {str(g["id"]): g for g in games}
    team_of = latest_team(stats)
    status = {str((r.get("player") or {}).get("id")): r for r in injuries}
    out_by_team = defaultdict(list)
    for r in injuries:
        pl = r.get("player") or {}
        if str(r.get("status") or "").lower() in BLOCKING_STATUS:
            out_by_team[str(pl.get("team_id"))].append(f"{pl.get('first_name', '')} {pl.get('last_name', '')}".strip())
    wanted = {}
    skipped = Counter()
    for r in props:
        if str(r.get("vendor") or "").lower() != vendor:
            continue
        market = policy["markets"].get(str(r.get("prop_type")))
        if market is None:
            continue
        over, under = prop_prices(r)
        if over is None or under is None or r.get("line_value") is None:
            skipped["not_two_sided"] += 1; continue
        gid, pid = str(r["game_id"]), str(r["player_id"])
        if pid not in team_of:
            skipped["no_player_history"] += 1; continue
        wanted[(gid, pid, market, float(r["line_value"]))] = (r, over, under)
    placeholders = []
    for (gid, pid) in sorted({(k[0], k[1]) for k in wanted}):
        g = tips.get(gid, {})
        placeholders.append({"min": "0:00", "pts": 0, "reb": 0, "ast": 0, "fg3m": 0, "pf": 0, "player": {"id": int(pid) if pid.isdigit() else pid},
                             "team": {"id": int(team_of[pid]) if team_of[pid].isdigit() else team_of[pid]},
                             "game": {"id": int(gid) if gid.isdigit() else gid, "date": str(g.get("date") or captured_at)[:10]}})
    feats = build_minutes(stats + placeholders, games) if placeholders else {}
    mm = minutes_model_from_recipe(recipe)
    rows = []
    for (gid, pid, market, line), (r, over, under) in wanted.items():
        it = feats.get((gid, pid))
        if it is None or it["rates"].get("threes") is None:
            skipped["insufficient_history"] += 1; continue
        pm = float(mm.predict_mean([it])[0])
        p_over_devig = power_devig_pair(over, under)[0]
        p_model, p_recal = threes_probability(recipe, pm, it["rates"]["threes"], line, p_over_devig)
        side = "over" if p_recal > p_over_devig else "under"
        p_side = p_recal if side == "over" else 1 - p_recal
        price = over if side == "over" else under
        ev = p_side * (1 + profit_per_unit(price)) - 1
        diff = abs(p_recal - p_over_devig)
        st = status.get(pid)
        pstat = str((st or {}).get("status") or "not_listed")
        qualified = diff >= policy["min_prob_diff"] and ev > 0 and pstat.lower() not in BLOCKING_STATUS
        mode = policy.get("mode", "shadow")
        stake = round(policy["bankroll"] * policy["stake_pct"], 2) if (qualified and mode == "real") else 0.0
        tip = str(tips.get(gid, {}).get("datetime") or "")
        row = {"ledger_version": LEDGER_VERSION, "recipe_sha256": recipe_sha, "market": market, "game_id": gid, "player_id": pid,
               "player_name": f"{(r.get('player') or {}).get('first_name', '')} {(r.get('player') or {}).get('last_name', '')}".strip() or None,
               "team_id": team_of.get(pid), "tip_utc": tip, "captured_at_utc": captured_at, "vendor": vendor,
               "vendor_updated_at": r.get("updated_at"), "line": line, "over_odds": over, "under_odds": under,
               "p_devig_over": round(p_over_devig, 5), "p_model_raw_over": round(p_model, 5), "p_model_over": round(p_recal, 5),
               "side": side, "side_price": price, "implied_prob_side": round(1 / (1 + profit_per_unit(price)), 5),
               "ev_per_unit": round(ev, 5), "prob_diff": round(diff, 5), "projected_minutes": round(pm, 2),
               "projected_threes": round(pm * it["rates"]["threes"], 3), "player_injury_status": pstat,
               "player_injury_note": (st or {}).get("description"), "teammates_out_or_doubtful": out_by_team.get(team_of.get(pid), []),
               "qualified": qualified, "mode": mode, "stake": stake}
        row["prediction_id"] = sha(f"{gid}|{pid}|{market}|{line}|{captured_at}")
        rows.append(row)
    return rows, dict(skipped)


def short_list(rows):
    q = sorted([r for r in rows if r["qualified"]], key=lambda r: -r["ev_per_unit"])
    lines = [f"**NBA props — {len(q)} qualified of {len(rows)} eligible** ({rows[0]['captured_at_utc'][:16]}Z, mode: {rows[0]['mode'] if rows else '-'})", ""]
    if not q:
        lines.append("No qualified bets. That is a normal outcome.")
    for r in q:
        stake = f"${r['stake']:.0f}" if r["stake"] else "shadow (no stake)"
        lines.append(f"- {r['player_name'] or r['player_id']} {r['market']} **{r['side']} {r['line']}** at {r['side_price']:+d} "
                     f"(model {r['p_model_over'] if r['side'] == 'over' else round(1 - r['p_model_over'], 4):.1%} vs fair "
                     f"{r['p_devig_over'] if r['side'] == 'over' else round(1 - r['p_devig_over'], 4):.1%}, EV {r['ev_per_unit']:+.1%}) — {stake}")
    lines += ["", "Unvalidated model: probabilities are research estimates, not guarantees. Check the price in your app before betting."]
    return "\n".join(lines)


# --------------------------------------------------------------------------- settle / report (pure)
def settle(predictions, closes, stats):
    box = {}
    for r in stats:
        g, pl = r.get("game") or {}, r.get("player") or {}
        if g.get("id") is not None and pl.get("id") is not None:
            box[(str(g["id"]), str(pl["id"]))] = r
    last_close = {}
    for c in closes:
        pid = c["prediction_id"]
        ct, tip = parse_utc(c.get("captured_at_utc")), parse_utc(c.get("tip_utc"))
        if ct and tip and ct <= tip and (pid not in last_close or ct > parse_utc(last_close[pid]["captured_at_utc"])):
            last_close[pid] = c
    out = []
    for p in predictions:
        b = box.get((p["game_id"], p["player_id"]))
        res = {"prediction_id": p["prediction_id"], "settled_at_utc": datetime.now(timezone.utc).isoformat()}
        if b is None:
            res["outcome"] = "void_or_pending"
        else:
            actual = float(b.get("fg3m") or 0)
            res["actual"] = actual
            if abs(actual - p["line"]) < 1e-9:
                res["outcome"] = "push"
            else:
                over_won = actual > p["line"]
                res["over_won"] = over_won
                res["outcome"] = "win" if over_won == (p["side"] == "over") else "loss"
            res["pnl"] = (p["stake"] * profit_per_unit(p["side_price"]) if res["outcome"] == "win" else
                          -p["stake"] if res["outcome"] == "loss" else 0.0)
        c = last_close.get(p["prediction_id"])
        if c and c.get("over_odds") is not None and c.get("under_odds") is not None:
            pc_over = power_devig_pair(c["over_odds"], c["under_odds"])[0]
            same_line = abs(float(c["line"]) - p["line"]) < 1e-9
            res["close_line"] = c["line"]; res["close_over_odds"] = c["over_odds"]; res["close_under_odds"] = c["under_odds"]
            res["close_same_line"] = same_line
            if same_line:
                pc_side = pc_over if p["side"] == "over" else 1 - pc_over
                res["clv_ev"] = round(pc_side * (1 + profit_per_unit(p["side_price"])) - 1, 5)
                res["close_p_devig_over"] = round(pc_over, 5)
        out.append(res)
    return out


def first_scan(predictions):
    """One prediction per (game, player, market): the earliest scan of the day is the one that counts."""
    keep = {}
    for p in sorted(predictions, key=lambda r: r["captured_at_utc"]):
        keep.setdefault((p["game_id"], p["player_id"], p["market"]), p)
    return list(keep.values())


def report(predictions, settlements):
    predictions = first_scan(predictions)
    sett = {s["prediction_id"]: s for s in settlements}
    rows = [(p, sett.get(p["prediction_id"])) for p in predictions]
    decided = [(p, s) for p, s in rows if s and s.get("outcome") in ("win", "loss")]
    rep = {"predictions": len(predictions), "qualified": sum(p["qualified"] for p in predictions), "decided": len(decided)}
    if decided:
        y = np.array([1.0 if s["over_won"] else 0.0 for p, s in decided])
        pm = np.clip([p["p_model_over"] for p, _ in decided], 1e-6, 1 - 1e-6); pk = np.clip([p["p_devig_over"] for p, _ in decided], 1e-6, 1 - 1e-6)
        ll = lambda q: -(y * np.log(q) + (1 - y) * np.log(1 - q))
        d = ll(pm) - ll(pk)
        rep["logloss_model_minus_market"] = round(float(d.mean()), 6)
        rep["logloss_ci"] = _cluster_ci(list(d), [p["game_id"] for p, _ in decided])
    q = [(p, s) for p, s in decided if p["qualified"]]
    clv = [(p, s) for p, s in rows if p["qualified"] and s and s.get("clv_ev") is not None]
    if clv:
        rep["qualified_clv_ev_mean"] = round(float(np.mean([s["clv_ev"] for _, s in clv])), 5)
        rep["qualified_clv_ci"] = _cluster_ci([s["clv_ev"] for _, s in clv], [p["game_id"] for p, _ in clv])
        rep["qualified_with_close"] = len(clv)
    if q:
        units = [profit_per_unit(p["side_price"]) if s["outcome"] == "win" else -1.0 for p, s in q]
        rep["qualified_decided"] = len(q); rep["qualified_unit_roi_pct"] = round(100 * float(np.mean(units)), 2)
        rep["qualified_hit_rate"] = round(sum(s["outcome"] == "win" for _, s in q) / len(q), 4)
    staked = [(p, s) for p, s in decided if p["stake"] > 0]
    if staked:
        rep["real_money"] = {"bets": len(staked), "staked": round(sum(p["stake"] for p, _ in staked), 2),
                             "pnl": round(sum(s["pnl"] for _, s in staked), 2)}
    rep["shadow_success_criteria"] = ">= 300 qualified with close, CLV mean > 0 with CI lower > 0, and log-loss model-minus-market CI upper < 0"
    return rep


def _cluster_ci(values, groups, reps=2000, seed=7):
    by = defaultdict(list)
    for v, g in zip(values, groups):
        by[g].append(v)
    arr = [np.array(v) for v in by.values()]
    if len(arr) < 10:
        return None
    rng = np.random.default_rng(seed)
    b = [np.concatenate([arr[i] for i in rng.integers(0, len(arr), len(arr))]).mean() for _ in range(reps)]
    return [round(float(np.percentile(b, 2.5)), 5), round(float(np.percentile(b, 97.5)), 5)]


# --------------------------------------------------------------------------- API glue
def api_get(session, path, params, key, tries=6):
    for attempt in range(tries):
        r = session.get(API + path, params=params, headers={"Authorization": key}, timeout=40)
        if r.status_code in (429, 500, 502, 503, 504) and attempt < tries - 1:
            time.sleep(min(60, 5 * (attempt + 1))); continue
        r.raise_for_status()
        return r.json()


def paged(session, path, params, key, max_pages=400):
    out, cursor = [], None
    for _ in range(max_pages):
        q = list(params) + [("per_page", "100")] + ([("cursor", str(cursor))] if cursor else [])
        body = api_get(session, path, q, key)
        out += body.get("data", [])
        cursor = (body.get("meta") or {}).get("next_cursor")
        if not cursor:
            break
        time.sleep(0.3)
    return out


def season_data(session, key, season):
    games = [g for g in paged(session, "/v1/games", [("seasons[]", str(season))], key) if not g.get("postseason")]
    finals = {str(g["id"]) for g in games if str(g.get("status")) == "Final"}
    from scripts.regular_season_filter import minutes
    stats = [r for r in paged(session, "/v1/stats", [("seasons[]", str(season))], key)
             if str((r.get("game") or {}).get("id")) in finals and minutes(r.get("min")) > 0]
    return games, stats


def main():
    import requests
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=("predict", "close", "settle", "report"))
    p.add_argument("--history-stats", action="append", default=[])
    p.add_argument("--history-games", action="append", default=[])
    p.add_argument("--recipe"); p.add_argument("--policy")
    p.add_argument("--predictions", action="append", default=[]); p.add_argument("--closes", action="append", default=[])
    p.add_argument("--settlements", action="append", default=[])
    p.add_argument("--season", type=int, default=2026)
    p.add_argument("--window-minutes", type=int, default=35)
    p.add_argument("--out-dir", required=True)
    a = p.parse_args()
    key = os.environ.get("BALLDONTLIE_API_KEY")
    now = datetime.now(timezone.utc); stamp = now.strftime("%Y%m%dT%H%M%SZ")
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    s = requests.Session()
    if a.command == "predict":
        recipe_text = Path(a.recipe).read_text(); recipe = json.loads(recipe_text)
        if recipe.get("status") != "frozen":
            sys.exit("recipe is not frozen")
        policy = json.loads(Path(a.policy).read_text())
        games, stats = season_data(s, key, a.season)
        upcoming = [g for g in games if parse_utc(g.get("datetime")) and now < parse_utc(g["datetime"]) <= now + timedelta(hours=20)]
        props = []
        for g in upcoming:
            props += api_get(s, "/nba/v2/odds/player_props", [("game_id", str(g["id"]))], key).get("data", [])
            time.sleep(0.3)
        injuries = paged(s, "/nba/v1/player_injuries", [], key)
        hist_s = [x for f in a.history_stats for x in read(f)]; hist_g = [x for f in a.history_games for x in read(f)]
        rows, skipped = build_predictions(hist_s + stats, hist_g + games, props, injuries, recipe, policy, now.isoformat(), sha(recipe_text))
        with (out / f"predictions_{stamp}.jsonl").open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        (out / f"injuries_{stamp}.json").write_text(json.dumps({"captured_at_utc": now.isoformat(), "rows": injuries}))
        (out / f"shortlist_{stamp}.md").write_text(short_list(rows) if rows else "No eligible props found for upcoming games.")
        print(json.dumps({"upcoming_games": len(upcoming), "props": len(props), "eligible": len(rows),
                          "qualified": sum(r["qualified"] for r in rows), "skipped": skipped}, indent=2))
    elif a.command == "close":
        preds = [x for f in a.predictions for x in read(f)]
        soon = defaultdict(list)
        for pr in preds:
            tip = parse_utc(pr.get("tip_utc"))
            if tip and now <= tip <= now + timedelta(minutes=a.window_minutes):
                soon[pr["game_id"]].append(pr)
        obs = []
        for gid, prs in soon.items():
            data = api_get(s, "/nba/v2/odds/player_props", [("game_id", gid)], key).get("data", [])
            for pr in prs:
                cands = [r for r in data if str(r.get("player_id")) == pr["player_id"] and str(r.get("vendor") or "").lower() == pr["vendor"]
                         and (r.get("market") or {}).get("type") == "over_under" and str(r.get("prop_type")) in ("threes", "threes_made")]
                for r in cands:
                    o, u = prop_prices(r)
                    obs.append({"prediction_id": pr["prediction_id"], "captured_at_utc": now.isoformat(), "tip_utc": pr["tip_utc"],
                                "line": float(r["line_value"]), "over_odds": o, "under_odds": u, "vendor_updated_at": r.get("updated_at")})
        with (out / f"closes_{stamp}.jsonl").open("w") as f:
            for r in obs:
                f.write(json.dumps(r) + "\n")
        print(json.dumps({"games_closing": len(soon), "observations": len(obs)}))
    elif a.command == "settle":
        preds = [x for f in a.predictions for x in read(f)]
        closes = [x for f in a.closes for x in read(f)]
        already = {x["prediction_id"] for f in a.settlements for x in read(f) if x.get("outcome") not in (None, "void_or_pending")}
        todo = [pr for pr in preds if pr["prediction_id"] not in already and parse_utc(pr.get("tip_utc"))
                and parse_utc(pr["tip_utc"]) < now - timedelta(hours=4)]
        games, stats = season_data(s, key, a.season) if todo else ([], [])
        rows = settle(todo, closes, stats)
        with (out / f"settlements_{stamp}.jsonl").open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(json.dumps({"settled": len(rows), "outcomes": dict(Counter(r["outcome"] for r in rows))}))
    else:
        preds = [x for f in a.predictions for x in read(f)]
        setts = {}
        for f in a.settlements:
            for x in read(f):
                if x.get("outcome") != "void_or_pending" or x["prediction_id"] not in setts:
                    setts[x["prediction_id"]] = x
        rep = report(preds, list(setts.values()))
        (out / f"report_{stamp}.json").write_text(json.dumps(rep, indent=2))
        print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
