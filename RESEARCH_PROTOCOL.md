# NBA V13 — Quantitative Research Protocol

## Mission
Aim for 200+ qualified regular-season bets and an aspirational 65% hit rate, without claiming either is achieved until independently verified. Optimize genuine predictive accuracy, then assess executable profitability and correlated bankroll exposure.

## Non-negotiable validation
1. Use chronological rolling-origin train/validation/test splits. Never randomly split games from the same time period.
2. Every feature must be available **before the game's tipoff**. Injury news, lineup confirmation, market odds, and narratives need source and observation timestamps. Never backfill pregame decisions using later information.
3. Keep a frozen, untouched holdout season. Compare every challenger with a simple Elo/home-court/rest baseline and an odds-informed baseline when timestamped odds exist.
4. Report picks, hit rate, Wilson 95% confidence intervals, Brier score, calibration, coverage, grade-by-grade accuracy, and results by spread, total, moneyline, and prop. Show actual bet prices separately from assumed -110 research prices.
5. Confidence 0–100 is a **ranking/quality score**, not an implied win probability. Calibrate it on out-of-sample games. A++/A+/A thresholds are provisional until monotonic performance by tier is demonstrated.
6. Keep complete pregame predictions, including passes and losing picks, in append-only timestamped logs. Do not change grades after tipoff.
7. Quantify uncertainty from injury status, rotation projections, model disagreement, and incomplete sources. If critical information is unavailable, lower confidence or pass.
8. Test proposed narrative signals against a baseline. Revenge, motivation, streaks, and locker-room reports do not receive numerical weight without prospective evidence.
9. Control multiple comparisons and overfitting: record all candidate factors, keep a challenger registry, and require sustained walk-forward improvement before promotion.
10. No claims of 65% performance, 200 qualified opportunities, or a competitive edge without a sufficient clean out-of-sample sample.

## Research layers
- Team strength: pace-adjusted offensive/defensive efficiency, shooting quality, turnover/rebound rates, home advantage, opponent strength, schedule and rest.
- Player availability: injuries, replacement rotations, expected minutes, usage, foul risk, minutes limits, and lineup stability.
- Matchups: defensive assignments, pick-and-roll coverage, switching, rim protection, size, and lineup-on-lineup interactions.
- Context: travel, back-to-backs, altitude, coaching changes, credible reports and verifiable narratives.
- Market: prioritize timestamped API bookmaker prices, prefer Caesars when present, otherwise closest available book; assumed -110 only as a transparent research fallback.
- Challenger models: regularized regression, gradient-boosted trees, hierarchical team/player models, and matchup-specific adjustments, subject to data sufficiency.

## Operational gates
**Gate 1:** Finish and verify complete historical game counts and dates. `BALLDONTLIE` 48-hour trial accounts are limited to 5 requests/minute, so pace conservatively.
**Gate 2:** Download historical player/advanced stats and validate API entitlements; do not assume historical injury, lineup, or odds snapshots exist.
**Gate 3:** Construct leak-free feature snapshots, outcomes, and baseline forecasts.
**Gate 4:** Run rolling backtests, compare challengers, and calibrate score/grade thresholds.
**Gate 5:** Start paper predictions with timestamped source snapshots, track all selections and misses, then evaluate readiness for real-money use.

## Data caveats
BALLDONTLIE currently documents historical opening odds only for the most recently completed and ongoing seasons where available. Live player props are not retained as historical snapshots. Store pregame data prospectively to enable future unbiased testing. The 65% target is an ambition, not an estimated or guaranteed success rate.
