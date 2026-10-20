# Pre-registration: 2024–25 rebounds procedure replication (v2 pipeline)

**Committed before any 2024–25 rebounds prop line is downloaded or graded.** As of this commit, no 2024–25 rebounds lines exist in the project: Prop Integrity v2 reported zero 2024 training coverage for rebounds.

## Origin of the hypothesis (exploratory, not evidence)
In the 2025–26 within-season v2 run (Prop Integrity v2, run 38642017481), the edge selector made 336 rebounds bets: actual-price ROI +6.88%, excess +7.02 pp over the de-vigged market, z = 2.59. That was the best of 5 markets × 2 selectors examined, so it is a selected result and needs a replication it played no part in.

## Exact procedure (no choices remain)
- Code: `scripts/backtest_prop_holdout_v2.py` at the commit that adds this file, unchanged. Run through `scripts/prereg_v2_market.py --market rebounds --selector edge`.
- Data:
  - 2024–25 regular-season pregame features: artifact `v13-2024-player-prop-features`, run 38614413903.
  - 2024–25 game index with tip times: `games.jsonl` in `v13-full-2024-threes-lines`, run 38614396898.
  - All 2024–25 two-sided opening props collected by `scripts/collect_season_prop_openings.py`.
- Line rule: fixed vendor priority; the earliest row opened strictly before tip; rows without both prices are excluded and counted.
- Split: the pipeline's built-in within-season 65/35 date split. The threshold is chosen on the inner 70/30 split of the training period, from the pipeline's rebounds edge grid [1, 1.5, 2, 2.5], requiring ≥ 40 validation bets, by its built-in rule.
- Primary endpoint: all rebounds edge-selected bets in the 2024–25 holdout (overs and unders together).
- Secondary (descriptive only): overs and unders separately.

## Frozen pass rule
**PASS only if all hold:**
- ≥ 150 decided bets;
- ROI at actual prices > 0;
- excess hit rate ≥ 2.0 pp AND z ≥ 2.0 under both multiplicative and power de-vig;
- ≥ 90% of matched rows carry two-sided prices.

**Otherwise FAIL** (or UNTESTABLE if price coverage is under 90%).

No other markets, selectors, thresholds, sides, line buckets or vendors may be evaluated as a rescue. A PASS licenses only a pre-registered forward test in 2026–27 at flat 1%. It does not authorize tiered staking.
