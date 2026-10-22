# Bucks at Wizards — V13 failure audit (2026-10-21)

**Classification:** losing wager; unvalidated research-driven recommendation; do not use this game to retune coefficients by hand.

## Verifiable final outcome
- Washington 128, Milwaukee 116 — total **244**.
- User's Caesars Under **239**, **−110**, **LOST**. User's receipt stake is **$140.91** (private wager details are **not** stored in this public repo).
- Earlier external projected total approximately **232.9**; error **+11.1** points. Do not mislabel an external projection as a fitted V13 probability.
- First half **107**, second half **137**; Q1 **48**, Q2 **59**, Q3 **76**, Q4 **61**.
- Source: normalized ESPN game `research_data/live_nba/2026-10-21/401909835.json` and the date-matched compressed complete ESPN summary/PBP archive. ESPN summary URL https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event=401909835

## Mechanism — observed, not assumed
- Estimated average possessions **104.58** (Milwaukee 104.12, Washington 105.04), based on **FGA − OREB + turnovers + 0.44 FTA**. Not optical tracking or exact end-to-end possession reconstruction.
- Historical prior-season pace figures **96.8** Bucks and **100.8** Wizards imply harmonic baseline **98.76**. Observed game is approximately **5.82 estimated possessions faster**.
- At actual combined scoring efficiency of **233.31 points / 100 team possessions** (MIL 110.92, WAS 122.39), those extra 5.82 possessions equate to about **13.58 points** as a retrospective accounting calculation, **not** an estimate of causality or an ex-ante prediction.
- Washington: **45-for-86 FG (52.3%)**, **6-for-28 from three (21.4%)**, **32-for-41 FT**, **70 points in the paint**, **12 OREB**, **13 total turnovers**. Washington scored 128 **without hot three-point shooting**.
- Milwaukee: **41-for-91 FG (45.1%)**, **15-for-41 three**, **19-for-23 FT**, **50 paint points**, **13 OREB**, **16 total turnovers**.
- Anthony Davis scored **34**; Trae Young **20**; Tyler Herro **24**. Rotations were very different from the previous-year baseline.
- ESPNs recorded **527 PBP entries** and **78 substitution events**. Raw events are archived. Validated full lineup-stint ratings, defender assignments and exact possession-clock measurements have **NOT** been produced.

## What went wrong in the process
1. **No trained and calibrated V13 live total model** supplied the selection. We substituted agreement among outside forecasts and an old-season mean. A mean difference of ~6 points was wrongly treated as evidence of positive expected value at −110.
2. **Pace uncertainty:** the prior-season pace baseline understated realized game tempo by roughly six possessions, enough to overturn the line. We never ran a sufficiently conservative pace sensitivity/stress test.
3. **Foul/paint risks:** against Milwaukee's new roster, Washington generated 41 FT attempts and 70 paint points; we did not explicitly model these volume/rim-pressure mechanisms.
4. **Roster/coaching structural break:** Milwaukee without Giannis and with new coach and additions, and Washington with its reworked Young/Davis group. Last-season raw team efficiency and pace were not transportable without a roster-continuity model.
5. **Insufficient market/model discipline:** we recommended an actual stake although opening-night uncertainty, lineup-sensitive minutes, and the absence of an out-of-sample calibrated total distribution should have resulted in a PASS.

## Committed safeguards
- Final-only NBA game snapshots, full raw ESPN PBP/boxscore compression, team+player summaries, substitutions, shot coordinates and checksums.
- Scheduled twice-daily replay of recent dates for all regular-season games; no access key required for ESPN public scoreboard.
- Prior-date rolling features for team tempo, shooting, opponent points allowed, free-throw pressure, rim scoring proxy, turnovers, rebounding, rest, known prior rotations and player game logs.
- A strict new model approval gate requiring validated market-specific OOS calibration, data as-of dates, fresh injuries/prices, lineup confidence, roster-churn sensitivity, and a conservative probability lower bound above break-even.
- Retain all the originally quoted market/forecast inputs; **never fit a new predictive parameter to this one loss** or insert its result into a pregame feature for that game.

## Not yet solved
- Historical seasons integrated into the **same** validated training and feature schema.
- Full per-lineup on-court possession stints and actual defending-player matchups. ESPN PBP shows play events and substitutions but **not** guaranteed defender assignments; advanced optical/tracking requires independent licensed/public coverage and validation.
- Empirically measured prediction residual distributions, walk-forward profit/loss on true timestamped lines, calibrated market-specific hit probabilities and model deployment. Until these exist, this is a research/archival pipeline, not an autonomous betting system.
