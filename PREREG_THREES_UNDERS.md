# Pre-registration: NBA three-point UNDER hypothesis (2024–25)

**Frozen before inspecting 2024–25 outcomes or calculating 2024–25 three-point betting performance.**

## Primary hypothesis
The historical 2025–26 three-point **unders** subset showed an exploratory positive actual-price return. The 2024–25 season is the untouched replication sample. No new filters, player subgroups, line buckets, bookmaker preferences, or thresholds may be chosen using its outcomes.

## Exact replication procedure
- Use the **legacy** `scripts/backtest_prop_holdout.py` pipeline and its original `normalize_lines`, `dataset`, `FEATURES`, `Ridge(alpha=25.)`, feature scaling and threshold-selection logic, without refitting parameters to the 2025–26 holdout.
- For 2024–25 only, split dates chronologically 65% training / 35% test, as the legacy script does.
- Use the original threes threshold grid **[0.5, 1.0, 1.5]**; choose the threshold on training rows by the original rule (minimum 60 training bets, maximum simulated training return, tie-break fewer bets).
- Fit the Ridge on 2024–25 training data only, generate projections for the final 35%, and select bets without viewing their outcomes.
- Use the exact original vendor priority and original line-selection logic. Preserve game/player/line/source IDs and timestamps for audit.
- The **primary endpoint is unders only** in the held-out 35%. Report overs and all bets as secondary results; no further subgroup optimization.
- Grade with the recorded side-specific **actual opening American odds**, not -110; only independently verified **pre-tip** opening records count.
- Compare observed hit rate against both multiplicative and power de-vig implied fair probability from the recorded two-sided prices.

## Frozen pass/fail rule
**Pass only if ALL hold:** unders n >= 150; unders ROI at actual prices > 0; observed minus de-vig expected hit rate >= 2 percentage points **under both** multiplicative and power de-vig; z >= 2.0 under both methods.

**Anything else fails** and retires this unders hypothesis. If data integrity or pre-tip verification is insufficient, classify as untestable, not as a pass. No re-slicing or threshold changes after outcomes are seen.

## Other guardrails
The overall 2025–26 three-point and assists strategies are retired pending independent evidence: threes 599–387, actual-price ROI -2.93%; assists 337–271, actual-price ROI -5.58%. Their -110 hypothetical results must not be used as evidence of profitability. No live wagers or A++ tier promotion on the basis of this hypothesis.

Created before opening 2024–25 result artifacts or computing 2024–25 bet outcomes.
