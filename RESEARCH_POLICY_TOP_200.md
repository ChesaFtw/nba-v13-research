# V13 Cross-Market Selection Policy — 2026-10-14

## Objective
Target approximately 200 total regular-season bets across spreads, totals, points, rebounds, assists, threes, and PRA. This is a research target, not a quota. Zero-bet nights are acceptable. Start bankroll $10,000; aspirational ending bankroll $15,000. Never tune on final holdout to force this result.

## Selection rules
1. Source actual historical opening thresholds for each market; assume -110 prices only for historical payout simulations, not as verified sportsbook odds.
2. Generate predictions with information available before the wager timestamp. Reject any leakage or uncertain data provenance.
3. Use rolling-origin training and nested validation for model choice, calibration, and selection thresholds. Final holdout must remain untouched until frozen.
4. Estimate calibrated probability for each eligible wager and conservative lower confidence bound. Rank across markets by out-of-sample expected value with uncertainty penalties; do not rank by raw edge in points.
5. Do not force per-market quotas or daily action. Annual ~200 target is soft. If insufficient qualifying opportunities exist, take fewer.
6. Pre-register confidence tiers: A = 1%, A+ = 2%, A++ = 3% of current bankroll per bet. Assign tiers using calibrated out-of-sample probabilities and conservative uncertainty bounds, with thresholds frozen before final holdout. Cap combined daily exposure at 6% and max 2 bets per game; correlated bets may require lower exposure. Compare against a flat 1% benchmark and report tier-by-tier ROI, win rate, sample size, drawdown and sensitivity. Never promote a wager to a larger stake solely to meet the $15,000 goal. Audit same-player, team, game, and overlapping prop correlations; avoid redundant positions.
7. Compare combined portfolio against no-bet, each market independently, a last-10 baseline, and the prior unfiltered strategy. Report wins/losses/pushes, net ROI, ending bankroll, drawdown, CLV only when actual time-stamped prices exist, and yearly consistency.
8. Record all eligible candidates and all rejected candidates with point-in-time predictions, line source, probability, calibration version, and reason for selection/rejection. Never choose selections with knowledge of outcomes.
9. Target ~200 bets/year, but do not force historical years to be positive or to end at $15,000. Report all failed tests and negative results.

## Current validation state
2025-26 spread holdout: 84-83 (50.30%), -7.13% simulated return at -110, 1% compounding.
2025-26 total holdout: 34-19 (64.15%), +12.38% simulated return at -110, 1% compounding; only 53 bets.
The prior combined portfolio selected 322 points props of 330 wagers due to fixed market priority, finishing $9,151.43 (-8.49%). This is NOT a fair best-of-all-markets selector.
High three-point prop performance requires integrity audit for line matching, timestamps, and leakage before deployment.

## Next implementation gate
Produce leakage-safe calibrated cross-market candidate ledger, select ~200 only from training/validation-defined thresholds, then evaluate the frozen combined portfolio on an untouched chronological holdout and additional historical seasons when actual line coverage is available.
