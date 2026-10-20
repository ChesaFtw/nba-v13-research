# Implementation handoff — independent audit, 2026-10-14

Read this whole file before changing code. It is ordered: do not start a phase until the
previous phase's gate has been reported to the owner. Never tune anything on a holdout
after seeing it, never invent prices or lines, and report failures as plainly as successes.

## Already done on branch `audit/claude-2026-10-14` (tested offline, NOT yet run on real data)

| File | What it does |
|---|---|
| `v13/pricing.py` | American-odds payout, implied and de-vigged probabilities, reads real prices from BALLDONTLIE records (`market.over_odds/under_odds`; `spread_home_odds`, `total_over_odds`, ...), strict `opened_at < tip` check using game `datetime`, Wilson intervals |
| `scripts/audit_prop_ledger.py` | Re-audits the EXISTING `holdout_2025.json` ledger: ROI at actual opening prices vs -110, pre-tip verification, duplicate/conflicting lines, excess hit rate vs de-vigged market, splits by side/line/price/vendor/month/edge |
| `scripts/backtest_prop_holdout_v2.py` | New prop backtest: actual prices (rows without prices excluded, never filled), pre-tip filter, threshold chosen on an inner validation slice with out-of-sample projections, start-of-day staking, edge selector and EV selector (Poisson/normal probability vs actual price), optional cross-season mode |
| `scripts/holdout_spreads_totals.py`, `run_spread_total_holdout.py`, `backtest_opening_spreads_totals.py` | Optional `--games` pre-tip filter and `--pricing actual`; legacy defaults unchanged so old numbers reproduce |
| `.github/workflows/prop-integrity-v2.yml` | Runs all of the above on existing artifacts (manual dispatch) |
| `tests/test_pricing_and_v2.py` | 12 tests, including a synthetic demonstration that -110 grading of half-point count lines produces ~60% hit rates with no skill |

The original `backtest_prop_holdout.py` is deliberately unchanged so earlier results can be reproduced.

## Phase 0 — Merge and run (no new modelling)

1. Merge the branch. Run `python -m pytest -q`; all tests must pass.
2. Dispatch **V13 Prop Integrity Audit v2**. If an artifact has expired, re-run the collector workflow that produced it first (pilot artifact retention is 30 days).
3. Add a one-off check that the spread sign is right: on the 2025 opening odds, `corr(-spread_home_value, actual_home_margin)` must be clearly positive (expect roughly 0.4–0.5). If it is negative, every spread result is reversed; stop and report.
4. Report to the owner, verbatim from the artifact JSON: for threes and assists, `overall` and `pregame_verified_only` from `ledger_audit_*`, the `by_side` and `by_price` tables, and the v2 within-season and cross-season holdout blocks for both selectors.

**Gate 0:** the 3PT result is treated as real only if ALL hold: pre-tip-verified ROI at actual prices > 0; `excess_hit_rate_vs_market` ≥ 0.02 with z ≥ 2.5; holds for both overs and unders; holds in the 2024→2025 cross-season run. Otherwise record it as a pricing artifact and move on. Do not search for a sub-slice that rescues it.

## Phase 1 — Data you cannot recreate later

5. Persist raw source data outside Actions artifacts: upload JSONL/Parquet to a GitHub Release per season (or a cloud bucket), with a manifest holding row counts, date range, SHA-256 and the collector commit.
6. Start a live price logger now (scheduled every 15 minutes on game days): BALLDONTLIE `/nba/v2/odds` and `/nba/v2/odds/player_props`, append-only, keyed by game, player, market, vendor, line, prices and `updated_at`. This is the only way to get closing lines and CLV going forward.
7. Probe and record opening game-odds and prop coverage (rows per game, share with both prices) for 2022-23, 2023-24 and 2024-25. Report gaps; never fill them.

## Phase 2 — Honest probability models

8. Replace "edge in points" with a probability per bet: negative binomial for threes, assists and rebounds (fit dispersion per stat on training data), normal with training-residual sigma for points and PRA. `p_over` in `backtest_prop_holdout_v2.py` is the Poisson starting point.
9. Add a minutes model before adding any other feature: predicted minutes from recent minutes, starter flag, teammates out (only from pre-tip injury reports), rest, and spread-based blowout risk. Then model per-minute rates with empirical-Bayes shrinkage.
10. Blend with the market in log-odds, fitted on validation seasons only: `logit(p) = a*logit(p_market_devigged) + b*logit(p_model) + c`. Report a, b, c. If b ≈ 0, the model adds nothing; say so.
11. Calibrate with isotonic regression on validation folds; on the holdout report reliability tables, Brier score and log loss versus the de-vigged market for each market and season.

**Gate 2:** the blended model must beat the de-vigged market's log loss on a holdout season it never saw. If it does not, stop before building a portfolio.

## Phase 3 — Selection, tiers and staking

12. Rank every candidate across markets by EV per unit at the actual price, using a conservative probability (5th percentile of a bootstrap over training residuals).
13. Limits before selection: one bet per player per game, two per game, 6% daily exposure, zero-bet nights allowed.
14. Tier map frozen before the holdout (quarter-Kelly at -110): A = 1% needs lower-bound p ≥ 0.543, A+ = 2% needs ≥ 0.562, A++ = 3% needs ≥ 0.581; for other prices use f* = (p·b − (1 − p)) / b and stake = min(f*/4, 3%).
15. Choose EV cutoffs on validation seasons to produce 100 / 150 / 200 / 250 / 300 bets; score the frozen cutoffs on the holdout. Report flat 1% and tiered staking side by side: ending bankroll, max drawdown, longest losing streak, per-tier hit rate with Wilson intervals, and 10,000-path Monte Carlo resampling of the holdout ledger.

## Phase 4 — Spreads and totals

16. Totals = projected possessions × (home + away points per possession). Possessions = FGA − ORB + TOV + 0.44·FTA, from opponent-adjusted rolling estimates with time decay and preseason priors.
17. Spreads = ridge-regressed opponent-adjusted team ratings, plus availability adjustments from pre-tip injury reports only, then a residual model against the opening line.
18. Use rolling-origin validation across all seasons with opening odds; untouched final season; report every season, winning or losing.

## Rules that apply throughout

- Every result file states: pricing source, line timestamp rule, train/validation/test dates, code commit, and every configuration that was tried (not only the winner).
- Never use `opened_at` without checking it is before the game's `datetime`.
- A missing price or line means the bet is excluded and counted, never assumed.
- Never report a hit rate without its count and 95% interval.
