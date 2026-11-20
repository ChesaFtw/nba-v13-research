# Preregistration: Injury-news timing and teammate player props

**Status: hypothesis only. Freeze this protocol before downloading or examining historical injury-report PDFs or injury-conditioned results.**

## Hypothesis
A newly reported OUT designation for a rotation player may increase teammates' subsequent minutes/usage before the market reprices teammate player props. The historical opening-line study tests an association; a **prospective executable-price study is required** before calling it a bettable edge.

## Population and exposure (fixed in advance)
- NBA regular-season games in 2025–26 with official NBA injury-report PDFs and trustworthy publication timestamps. Do not add seasons or restrict teams after observing outcomes.
- A **rotation player** has >=24.0 minutes per game across his last 10 completed regular-season appearances before the report, with at least 5 appearances; use strictly pre-report box scores. If fewer than 5, classify ineligible.
- A **new OUT event** is a player's first official status change from an explicitly non-OUT designation (available/probable/questionable/doubtful) to **OUT** in consecutive archived reports for that game. Players first appearing as OUT without an earlier status are ineligible. Only the first qualifying event per player-game counts.
- **News window:** the OUT report must be published >0 and <=24 hours before scheduled game tip. The teammate prop opening timestamp must be earlier than the OUT report publication timestamp. A line timestamp at/after the report is ineligible.
- **Bets:** OVER only on points, rebounds and assists props for other players on the same team and in the same game; no combined props, unders, or alternative lines. Use the original fixed vendor priority and earliest eligible two-sided opening quote per player-game-market; one bet per player-game-market even if multiple teammate OUT events.
- No filtering by player, market, position, odds, line value, projected role, number of OUT players, or historical profitability. Grade pushes as 0 P&L and exclude them from hit-rate z denominators; include staked amounts in ROI.

## Controls and scoring
- **Control:** the same eligible teammate player-game-market openings on nights with no qualifying teammate OUT event in the 24h pre-tip window. Match on market, price bucket and pregame window using fixed, non-outcome-dependent bins; report raw and stratified control statistics. Controls are descriptive and cannot substitute for the primary market-based test.
- Grade outcomes using official box scores and **recorded side-specific opening American odds** only. A missing price, missing timestamp, unknown prior injury status, or post-tip opening is excluded with counts reported.
- Compare observed outcomes to fair implied probability from **both multiplicative and power de-vig** of the recorded two-sided opening prices. Report n, win/loss/push, hit rate, actual-price ROI, excess percentage points and z under both methods; cluster uncertainty by game and report a game-cluster bootstrap CI as a sensitivity analysis.
- Record each source PDF URL, report publication timestamp, prior and new status, quote vendor, quote timestamp, both odds, game ID and player ID. Audit opening timestamps against tip and document the vendor's ambiguous 'opened_at' semantics.

## Frozen primary pass/fail
**Pass only if ALL hold:** at least **300 graded non-push eligible bets**; ROI at actual prices >0; excess hit rate >=2.0 percentage points and z >=2.5 under **both multiplicative and power de-vig**. Otherwise **FAIL** (or UNTESTABLE if timestamp/provenance requirements cannot be established). Do not re-slice or adjust thresholds after inspecting outcomes.

**Execution gate, independent of the historical pass:** verify in a timestamped prospective odds log that the identical quote remained available after the OUT report and before suspension/repricing, with a documented availability window and source. Without this, classify any positive historical result as a *non-executable opening-line association*, not a trading strategy.

This preregistration authorizes data collection and research only. It does not authorize wagers, A++ staking or market promotion.
