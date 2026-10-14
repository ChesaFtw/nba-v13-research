# V13 Challenger Gate — 2026-10-14

## Audit
- 2025 regular-season pilot GitHub Actions run 38584634800: successful.
- scripts/regular_season_benchmark.py compares rolling projections and Ridge on a chronological holdout; it does **not** score actual betting lines.
- scripts/walk_forward_prop_mae.py evaluates expanding-window Ridge versus last-10 on pregame features, but is explicitly MAE-only.
- PROJECT_STATUS.md is stale; do not use it as a source of current model readiness.
- The workflow artifact has a 30-day retention policy. Historical data availability beyond artifacts is not yet verified.

## Next implementation gates
1. Download/verify artifacts for each season, completeness manifests, date ranges, game counts, and pregame availability. Fail closed on missing or inconsistent provenance.
2. Introduce an immutable market-line table keyed by game/player, market, selection, threshold, source, and observed_at. A default **-110 price is permitted**, but missing thresholds must never be invented.
3. Build a single prediction ledger with game ID, event time, prediction time, model version, features provenance, probability, grade, actual market threshold, result and price provenance.
4. Implement leakage-safe rolling-origin comparisons on identical folds: rolling baseline, Ridge/ElasticNet, boosting, and equal-weight ensemble. Tune within training folds only; freeze holdout.
5. Report per-market selection hit rate, Brier/log loss, calibration, number of picks, MAE, and -110 hypothetical ROI/drawdown where actual thresholds exist. Do not equate MAE with ATS/prop hit rate.
6. Add injury, minutes, opponent matchup and uncertainty features only after timestamp checks and ablation evidence.
7. Only after holdout validation, run timestamped preseason shadow and regular-season paper rankings. Never count shadow predictions as placed bets.

## Decision rules
- Optimize independently validated hit rate and reliability; assume -110 for provisional pricing.
- Top 10 are candidates, not forced wagers. A 80-89 = 1%, A+ 90-94 = 2%, A++ 95-100 = max 2%; below 80 pass.
- Actual Caesars Kansas market line/price must be checked before any real-money approval.
- Do not claim 65% performance, 150-200 qualified bets, or validated profitability without untouched test evidence.
