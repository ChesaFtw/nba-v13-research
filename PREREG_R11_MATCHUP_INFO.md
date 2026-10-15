# Pre-registration R11: do matchup and career-stage features add information beyond the opening price?

**Committed before any R11 statistic is computed on real data.** None of these features has been computed on real data before.

## Question
For each (market, feature) pair below: does adding the feature to the de-vigged opening price reduce **out-of-sample log loss** for over/under outcomes?

This is an *information* test. It is not a betting rule.

## Frozen registry (`scripts/feature_info_test.py`, `REGISTRY`)

| Market | Features |
|---|---|
| Rebounds | `height_diff`, `opp_missed_fg`, `own_missed_fg`, `height_x_oppmiss`, `opp_fg3a_rate` |
| Points | `h2h_resid_points`, `years_since_draft`, `yrs_x_b2b` |
| Assists | `h2h_resid_assists`, `years_since_draft`, `yrs_x_b2b` |

Feature definitions are in `v13/matchup_features.py`. Every feature uses only games dated strictly before the game. `years_since_draft` is a career-stage proxy, because BALLDONTLIE has no birth dates. Head-to-head history uses every supplied season of box scores (2023-24 onward), shrunk with k = 5.

## Data
- Gated 2025-26 opening props: the `bdl_opening_player_props` gate.
- One line per player-game-market, by fixed vendor priority.
- Played-minutes box scores only (DNPs are void); pushes are excluded.

## Design
- **Train:** dates before 2026-02-01. **Test:** dates on or after 2026-02-01.
- **Baseline model:** logistic regression of the over outcome on logit(p_market, power de-vig).
- **Challenger:** the baseline plus the standardized feature.
- **Uncertainty:** game-cluster bootstrap (1,000 resamples) of the test log-loss difference.
- **Multiple testing:** Benjamini–Hochberg at q = 0.10 across all tested pairs.

## Frozen pass rule (per pair)
**PASS only if all hold:**
- the test log-loss difference is < 0;
- the pair is BH-significant;
- the feature's coefficient has the same sign when fitted on each half of the training period.

**Anything else is a FAIL.** No feature, threshold, split or market may be added after results are seen. Any new idea goes into a new pre-registration.

## What a PASS licenses
2025-26 prop prices have been used for earlier, different hypotheses, but never with these features. So a PASS is single-season evidence of information. It licenses only:
1. a pre-registered betting rule built from the passing feature;
2. a 2026-27 shadow test of that rule, graded at logger-captured prices and on CLV.

It does not license staking.
