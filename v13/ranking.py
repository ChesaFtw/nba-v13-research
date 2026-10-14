"""Universal cross-market ranking: every wager scored on one scale.

Why this works across markets: a 1.5-assist edge and a 4-point spread edge are not
comparable, but "probability of winning at this price, versus what the price
requires" is. Each market's raw model probability is recalibrated AND blended with
the de-vigged market probability by a logistic fit on strictly earlier data:

    logit(p) = c + a * logit(p_market) + b * logit(p_model)

If a market's model adds nothing, b -> 0, p -> p_market, expected value -> minus
the vig, and that market simply stops producing selections. No quotas needed.

Ranking key = EV at a conservative probability p_lo (one-sided 90% bound from the
fit's parameter uncertainty), so thin or unstable markets are penalised.

Candidate schema (one JSON object per side per line; both sides should be present):
  candidate_id, season, date (YYYY-MM-DD), game_id, player_id (or null), market,
  side, line, price (American, this side), opp_price (other side, or null),
  p_model (model P(win | no push)), p_push (optional, default 0),
  outcome ("win"/"loss"/"push"; used only for fitting on PAST rows and for grading)
"""
import math
from collections import Counter, defaultdict

import numpy as np

from v13.pricing import implied_probability, no_vig_pair, profit_per_unit, wilson

EPS = 1e-4


def logit(p):
    p = min(max(p, EPS), 1 - EPS)
    return math.log(p / (1 - p))


def sigmoid(z):
    return 1 / (1 + math.exp(-z))


def market_probability(c):
    if c.get("opp_price") is None or c.get("price") is None:
        return None
    return no_vig_pair(c["price"], c["opp_price"])[0]


def _features(c, use_market):
    x = [1.0, logit(c["p_model"])]
    if use_market:
        x.append(logit(c["p_market"]))
    return x


class Calibrator:
    """Weighted logistic regression (Newton) with a small ridge; keeps the covariance."""

    def __init__(self, ridge=1.0, min_rows=200):
        self.ridge, self.min_rows = ridge, min_rows
        self.ok = False

    def fit(self, rows):
        rows = [r for r in rows if r["outcome"] in ("win", "loss")]
        self.use_market = bool(rows) and all(r.get("p_market") is not None for r in rows)
        if len(rows) < self.min_rows:
            return self
        X = np.array([_features(r, self.use_market) for r in rows])
        y = np.array([1.0 if r["outcome"] == "win" else 0.0 for r in rows])
        # Both sides of a line are usually present and perfectly anti-correlated: weight 0.5 each.
        w = np.full(len(rows), 0.5)
        beta = np.zeros(X.shape[1])
        if self.use_market:
            beta[2] = 1.0  # start at "trust the market"
        pen = np.eye(X.shape[1]) * self.ridge
        pen[0, 0] = 0
        for _ in range(50):
            p = 1 / (1 + np.exp(-(X @ beta)))
            g = X.T @ (w * (y - p)) - pen @ beta
            H = X.T @ (X * (w * p * (1 - p))[:, None]) + pen
            step = np.linalg.solve(H, g)
            beta += step
            if np.max(np.abs(step)) < 1e-8:
                break
        self.beta, self.cov, self.n, self.ok = beta, np.linalg.inv(H), len(rows), True
        return self

    def predict(self, c, z=1.2816):
        """(p, p_lo). z=1.2816 -> one-sided 90% lower bound."""
        if not self.ok:
            return None, None
        x = np.array(_features(c, self.use_market))
        eta = float(x @ self.beta)
        se = float(math.sqrt(max(x @ self.cov @ x, 0.0)))
        return sigmoid(eta), sigmoid(eta - z * se)

    def describe(self):
        if not self.ok:
            return {"status": "insufficient rows"}
        names = ["intercept", "b_model"] + (["a_market"] if self.use_market else [])
        se = np.sqrt(np.diag(self.cov))
        return {"n": self.n, **{k: round(float(v), 4) for k, v in zip(names, self.beta)},
                **{f"se_{k}": round(float(s), 4) for k, s in zip(names, se)}}


def prepare(candidates):
    out = []
    for c in candidates:
        c = dict(c)
        c["p_market"] = market_probability(c)
        c.setdefault("p_push", 0.0)
        out.append(c)
    return sorted(out, key=lambda c: (c["date"], str(c["game_id"]), c["market"], str(c.get("player_id")), c["side"]))


def walk_forward_score(candidates, refit="month", grouping="market", **cal_kwargs):
    """Score every candidate with calibrators fitted only on candidates dated before its period.

    grouping="market": one calibrator per (market, model_version) [default].
    grouping="pooled": one calibrator for all markets (Experiment B comparison).
    """
    key = (lambda c: (c["market"], c.get("model_version"))) if grouping == "market" else (lambda c: ("pooled", "all"))
    cands = prepare(candidates)
    period = (lambda d: d[:7]) if refit == "month" else (lambda d: d)
    periods = sorted({period(c["date"]) for c in cands})
    scored, fits = [], {}
    for per in periods:
        start = min(c["date"] for c in cands if period(c["date"]) == per)
        past = [c for c in cands if c["date"] < start]
        bymarket = defaultdict(list)
        for c in past:
            bymarket[key(c)].append(c)
        cals = {m: Calibrator(**cal_kwargs).fit(rows) for m, rows in bymarket.items()}
        fits[per] = {f"{m}|{v}": cal.describe() for (m, v), cal in cals.items()}
        for c in cands:
            if period(c["date"]) != per:
                continue
            cal = cals.get(key(c))
            p, p_lo = cal.predict(c) if cal else (None, None)
            if p is None:
                continue
            b = profit_per_unit(c["price"])
            no_push = 1 - c["p_push"]
            scored.append({**c, "p": p, "p_lo": p_lo,
                           "ev": no_push * (p * b - (1 - p)),
                           "ev_lo": no_push * (p_lo * b - (1 - p_lo))})
    return scored, fits


def kelly_tier(c, divisor=4.0):
    """Stake fraction from quarter-Kelly on the conservative probability, bucketed to 0/1/2/3%."""
    b = profit_per_unit(c["price"])
    f = (c["p_lo"] * b - (1 - c["p_lo"])) / b
    q = f / divisor
    if q >= 0.025: return 0.03, "A++"
    if q >= 0.015: return 0.02, "A+"
    if q >= 0.0075: return 0.01, "A"
    return 0.0, "pass"


def select(scored, tau, per_game=2, per_player_game=1, daily_exposure=0.06, unit=0.01):
    """Greedy by ev_lo within each day, subject to correlation and exposure limits."""
    chosen = []
    byday = defaultdict(list)
    for c in scored:
        byday[c["date"]].append(c)
    for day in sorted(byday):
        games, player_games, game_markets = Counter(), Counter(), set()
        exposure = 0.0
        for c in sorted(byday[day], key=lambda c: -c["ev_lo"]):
            if c["ev_lo"] < tau:
                break
            stake, tier = kelly_tier(c)
            stake = max(stake, unit)
            gm = (str(c["game_id"]), c["market"], str(c.get("player_id")))
            if gm in game_markets or games[str(c["game_id"])] >= per_game:
                continue
            if c.get("player_id") is not None and player_games[(str(c["game_id"]), str(c["player_id"]))] >= per_player_game:
                continue
            if exposure + stake > daily_exposure + 1e-9:
                continue
            games[str(c["game_id"])] += 1
            if c.get("player_id") is not None:
                player_games[(str(c["game_id"]), str(c["player_id"]))] += 1
            game_markets.add(gm)
            exposure += stake
            chosen.append({**c, "tier": tier, "tier_stake": stake})
    return chosen


STAKING = ("flat", "tiered", "capped", "half")


def stake_fraction(b, staking, unit=0.01):
    """flat: 1% | tiered: 1/2/3% | capped: tiers capped at 2% | half: 0.5/1/1.5%."""
    if staking == "flat":
        return unit
    t = b["tier_stake"]
    if staking == "tiered":
        return t
    if staking == "capped":
        return min(t, 0.02)
    if staking == "half":
        return t / 2
    raise ValueError(staking)


def simulate(bets, staking="flat", bankroll=10000.0, unit=0.01):
    """Day-start staking: every bet on a date is sized from the bankroll before that date's results."""
    led, peak, streak, worst_streak = [], bankroll, 0, 0
    byday = defaultdict(list)
    for b in bets:
        byday[b["date"]].append(b)
    for day in sorted(byday):
        start = bankroll
        for b in byday[day]:
            stake = round(start * stake_fraction(b, staking, unit), 2)
            o = b["outcome"]
            pnl = round(stake * (profit_per_unit(b["price"]) if o == "win" else -1 if o == "loss" else 0), 2)
            bankroll = round(bankroll + pnl, 2)
            peak = max(peak, bankroll)
            streak = streak + 1 if o == "loss" else (0 if o == "win" else streak)
            worst_streak = max(worst_streak, streak)
            led.append({**b, "stake": stake, "pnl": pnl, "bankroll": bankroll,
                        "drawdown_pct": round(100 * (peak - bankroll) / peak, 3)})
    return summarize(led, worst_streak)


def summarize(led, worst_streak=None):
    c = Counter(x["outcome"] for x in led)
    n = c["win"] + c["loss"]
    staked = sum(x["stake"] for x in led)
    tiers = {}
    for t in ("A", "A+", "A++", "pass"):
        sub = [x for x in led if x.get("tier") == t]
        if sub:
            cc = Counter(x["outcome"] for x in sub); nn = cc["win"] + cc["loss"]
            st = sum(x["stake"] for x in sub)
            tiers[t] = {"bets": len(sub), "hit_rate": round(cc["win"] / nn, 4) if nn else None,
                        "hit_95ci": wilson(cc["win"], nn),
                        "mean_breakeven": round(sum(implied_probability(x["price"]) for x in sub) / len(sub), 4),
                        "roi_pct": round(100 * sum(x["pnl"] for x in sub) / st, 2) if st else None}
    markets = {}
    for m in sorted({x["market"] for x in led}):
        sub = [x for x in led if x["market"] == m]
        cc = Counter(x["outcome"] for x in sub); st = sum(x["stake"] for x in sub)
        markets[m] = {"bets": len(sub), "wins": cc["win"], "losses": cc["loss"],
                      "roi_pct": round(100 * sum(x["pnl"] for x in sub) / st, 2) if st else None}
    return {"bets": len(led), "wins": c["win"], "losses": c["loss"], "pushes": c["push"],
            "hit_rate": round(c["win"] / n, 4) if n else None, "hit_95ci": wilson(c["win"], n),
            "roi_on_turnover_pct": round(100 * sum(x["pnl"] for x in led) / staked, 2) if staked else None,
            "ending_bankroll": led[-1]["bankroll"] if led else 10000.0,
            "max_drawdown_pct": max((x["drawdown_pct"] for x in led), default=0.0),
            "longest_losing_streak": worst_streak, "by_tier": tiers, "by_market": markets, "ledger": led}


def tau_for_target(scored, target, lo=-0.05, hi=0.30, iters=40, **kw):
    """Smallest EV_lo cutoff whose selection count on `scored` does not exceed target."""
    if len(select(scored, lo, **kw)) <= target:
        return lo
    for _ in range(iters):
        mid = (lo + hi) / 2
        if len(select(scored, mid, **kw)) > target:
            lo = mid
        else:
            hi = mid
    return hi


def monte_carlo(ledger, staking="flat", paths=10000, seed=7, unit=0.01, bankroll=10000.0):
    """Bootstrap whole betting days from a holdout ledger; returns ending-bankroll distribution."""
    rng = np.random.default_rng(seed)
    byday = defaultdict(list)
    for b in ledger:
        byday[b["date"]].append(b)
    days = list(byday.values())
    if not days:
        return None
    ends, dds = [], []
    for _ in range(paths):
        bank, peak, mdd = bankroll, bankroll, 0.0
        for i in rng.integers(0, len(days), len(days)):
            start = bank
            for b in days[i]:
                o = b["outcome"]
                bank += start * stake_fraction(b, staking, unit) * (profit_per_unit(b["price"]) if o == "win" else -1 if o == "loss" else 0)
            peak = max(peak, bank); mdd = max(mdd, (peak - bank) / peak)
        ends.append(bank); dds.append(mdd)
    ends, dds = np.array(ends), np.array(dds)
    return {"paths": paths, "p5": round(float(np.percentile(ends, 5))), "p50": round(float(np.median(ends))),
            "p95": round(float(np.percentile(ends, 95))), "prob_ge_15000": round(float((ends >= 15000).mean()), 4),
            "prob_below_start": round(float((ends < bankroll).mean()), 4),
            "median_max_drawdown_pct": round(100 * float(np.median(dds)), 2),
            "p95_max_drawdown_pct": round(100 * float(np.percentile(dds, 95)), 2)}


def reliability(scored, bins=10):
    """Calibration table plus Brier/log loss of blended p vs de-vigged market, decided rows only."""
    rows = [c for c in scored if c["outcome"] in ("win", "loss")]
    if not rows:
        return None
    table = defaultdict(list)
    for c in rows:
        table[min(bins - 1, int(c["p"] * bins))].append(c)
    out = {"bins": [{"bin": k, "n": len(v), "mean_p": round(sum(c["p"] for c in v) / len(v), 4),
                     "hit_rate": round(sum(c["outcome"] == "win" for c in v) / len(v), 4)} for k, v in sorted(table.items())]}
    y = [1.0 if c["outcome"] == "win" else 0.0 for c in rows]
    def ll(ps): return -sum(yy * math.log(max(p, EPS)) + (1 - yy) * math.log(max(1 - p, EPS)) for yy, p in zip(y, ps)) / len(y)
    out["n"] = len(rows)
    out["brier_model"] = round(sum((c["p"] - yy) ** 2 for c, yy in zip(rows, y)) / len(y), 5)
    out["logloss_model"] = round(ll([c["p"] for c in rows]), 5)
    mk = [c for c in rows if c.get("p_market") is not None]
    if len(mk) == len(rows):
        out["brier_market"] = round(sum((c["p_market"] - yy) ** 2 for c, yy in zip(rows, y)) / len(y), 5)
        out["logloss_market"] = round(ll([c["p_market"] for c in rows]), 5)
    return out


def parametric_monte_carlo(bets, shrink=1.0, staking="flat", paths=5000, seed=11, unit=0.01, bankroll=10000.0):
    """Sensitivity to weaker-than-estimated edge, keeping the real bet/price/day structure.

    True win probability of each bet = p_market + shrink * (p - p_market):
    shrink=1 -> our calibrated model is exactly right; 0.5 -> half the estimated edge is real;
    0 -> the market is right (expected loss = the vig). Outcomes are re-drawn; pushes ignored.
    """
    rows = [b for b in bets if b.get("p_market") is not None and b.get("p") is not None]
    if not rows:
        return None
    rng = np.random.default_rng(seed)
    byday = defaultdict(list)
    for b in rows:
        byday[b["date"]].append(b)
    days = [byday[d] for d in sorted(byday)]
    ends, dds = [], []
    for _ in range(paths):
        bank, peak, mdd = bankroll, bankroll, 0.0
        for day in days:
            start = bank
            for b in day:
                p_true = min(max(b["p_market"] + shrink * (b["p"] - b["p_market"]), 0.0), 1.0)
                win = rng.random() < p_true
                bank += start * stake_fraction(b, staking, unit) * (profit_per_unit(b["price"]) if win else -1)
            peak = max(peak, bank); mdd = max(mdd, (peak - bank) / peak)
        ends.append(bank); dds.append(mdd)
    ends, dds = np.array(ends), np.array(dds)
    return {"shrink": shrink, "staking": staking, "mean_end": round(float(ends.mean())),
            "p50": round(float(np.median(ends))), "prob_ge_15000": round(float((ends >= 15000).mean()), 4),
            "prob_below_start": round(float((ends < bankroll).mean()), 4),
            "median_max_drawdown_pct": round(100 * float(np.median(dds)), 2),
            "p95_max_drawdown_pct": round(100 * float(np.percentile(dds, 95)), 2)}
