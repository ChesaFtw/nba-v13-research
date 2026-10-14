# V13 spread and total model research program

## Why current model needs a redesign
The current baseline predicts margin and total from last-ten points scored and allowed. This confounds pace and efficiency, ignores opponent strength and roster availability, and treats market prices as an afterthought. Its selective spread holdout was 84-83 (-7.13% at assumed -110), whereas selective totals were 34-19 (+12.38%) on just 53 bets. Neither is proof of a repeatable edge.

## Pre-game features to engineer
- Possessions and pace: estimate from FGA, FTA, offensive rebounds, turnovers. Rolling 5/10/20 games with opponent adjustment, shrink early-season values toward prior-season baseline.
- Offense/defense: points per 100 possessions, four factors (effective FG%, turnover%, offensive rebound%, free throw rate), three-point volume and shot quality when sourced.
- Rotation and availability: injury reports with publication timestamps; projected starters, top-8 minutes, missing usage, replacements and on/off only when point-in-time snapshots exist. Reject unverifiable retrospective lineup assumptions.
- Rest/travel: days since last game, back-to-back, three-in-four, travel distance, time zone, altitude, home/away and early start. Verify schedule timestamps.
- Opponent and tactical matchups: transition pace, rim vs three-point shot mix, offensive rebounding, foul drawing, defensive scheme when measured; account for opponent strengths and sample sizes.
- Context: schedule density, recent overtime, roster transactions and coaching changes only with pre-game timestamps.
- Market baseline: historical opening spread/total and bookmaker; learn residual against market rather than projecting raw outcomes alone. The market is a strong baseline. Ensure the opening quote precedes the prediction time.
- Distinct totals model: predict possessions and points-per-possession separately, then calibrate total distribution, rather than adding two naive team point projections.
- Distinct spreads model: opponent-adjusted net rating, player availability and home-court, anchored to market spread. Test whether edge persists after injury and price adjustments.

## Research methodology
1. Verify possession and injury data availability; never imply data exists without a source.
2. Pre-register three baselines: opening market (zero bets), rolling team model, and opponent/rest challenger.
3. Train chronologically across earlier seasons, nested validation for hyperparameters and threshold selection; reserve latest season for untouched test. Use embargo or strict as-of joins where needed.
4. Compare Ridge, gradient boosting and shrinkage-to-market; benchmark probability calibration with reliability plots, Brier score and log loss. Avoid unbounded complexity.
5. Grade actual sourced opening lines; historical -110 assumed payouts are research only. Pushes, vig and line timing must be explicit.
6. Produce spread and total performance separately by edge decile, side, pace regime, injury uncertainty, rest differential, and season. Require adequate samples and confidence intervals; avoid slicing until a winner appears.
7. Compete against player props for approximately 200 total annual wagers. Use confidence tiers A 1%, A+ 2%, A++ 3% only when calibrated validation supports them. Limit correlated game exposure and maximum daily exposure.
8. Freeze thresholds before final holdout, publish both failures and wins. No guarantees of profitability or $15k.

## Immediate experiments
- Fit opening-line-anchored spread and total residual models with opponent-adjusted efficiency and rest features.
- Compare independent totals pace × efficiency against points-for/against baseline.
- Validate selection by calibration and bet-count stability, not only highest retrospective ROI.
