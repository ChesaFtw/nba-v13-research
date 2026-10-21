# Opening-night audit — 2026-10-20

## Confirmed results
- BOS @ DET: Jayson Tatum 4 rebounds in 37:49; Over 8.5 at -121, $100 stake, loss -$100. NBA box score: https://www.nba.com/game/bos-vs-det-0022600001/box-score
- OKC @ SAS: 116-104 (220 points); Under 231.5 at -110, $100 stake, win +$90.91. NBA box score: https://www.nba.com/spurs/game/0022600003-thunder-vs-spurs-san-antonio-tx-10-20-2026/box-score
- Combined realized P/L: -$9.09. Do not include unconfirmed Knicks parlay.

## Diagnosis, without hindsight fitting
- Tatum: minutes alone cannot explain four rebounds; audit rebound chances, on-court rebound share, opponent shot mix, teammate lineup competition, and post-injury/role uncertainty. These are hypotheses, not verified causal explanations.
- OKC-SAS: winning under by 11.5 is not proof of predictive edge; audit pregame projected pace and efficiency, pregame quote timestamp, closing line, and play-by-play pace versus expectations.
- Preserve original pregame predictions and inputs; do not retroactively invent probability forecasts or closing lines.

## Research gate for proposed changes
1. Pre-bet: player/team eligibility, confirmed lineup/injury and projected minutes; reject when uncertain or stale.
2. Quantify per-minute rebound opportunities and player share conditional on likely five-man units; compare to minutes-only and sportsbook baselines.
3. Use shrinkage for role changes and opening-week uncertainty. Do not treat one-game outcomes as training signals with special weights.
4. Evaluate rolling chronological out-of-sample seasons, grouped by market, price, and role-change status; report calibration, log loss, hit rate, ROI, and sample sizes.
5. Reject any new betting threshold without out-of-time holdout improvement and adequate samples; no promotion of model based on this two-bet audit.
6. Reconcile each placed ticket and settlement with actual odds, stake, and final box score; capture closing odds separately for CLV.
7. Keep Knicks-opposition wagers excluded per user preference, independently of quantitative model predictions.

Code: `research/opening_night_postgame_audit.py` records the two verified wagers and rejects model-unsupported recommendations. It does not change production prediction weights or establish a validated edge.
