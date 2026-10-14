# Frozen Experiment A — 2024–25 three-point unders replication

**Verdict: FAIL — retire this hypothesis.** The preregistration was committed before examining 2024–25 outcomes.

- Workflow: https://github.com/ChesaFtw/nba-v13-research/actions/runs/38641909154
- Artifact: `v13-prereg-threes-unders-2024` (`threes_unders_2024.json`, artifact ID 11929924082).
- 65/35 chronological split; holdout begins **2025-02-13**; original training-selected threshold **0.5**.
- Pre-tip verified: **1,272/1,272** graded bets (100%).
- **Primary unders:** 394 bets, 232 wins, 162 losses, 58.88% hit rate, **-3.55% actual-price ROI**.
- Multiplicative de-vig: expected hit 57.03%; excess +1.86 percentage points, z=0.75.
- Power de-vig: expected hit 57.86%; excess +1.02 percentage points, z=0.41.
- **Secondary overs:** 878 bets, 524–354, **-3.62% actual-price ROI**.
- **All threes:** 1,272 bets, 756–516, **-3.60% actual-price ROI**.

The frozen requirements (unders n>=150, ROI>0, excess>=2pp AND z>=2 under both de-vig methods) were not met. **No post-hoc reslicing, threshold optimization, or live deployment of this hypothesis.**

The legacy all-bets report showed 59.43% hit rate and positive hypothetical -110 performance; the negative actual-price ROI shows why hypothetical pricing is not actionable.

Earlier 2025–26 exploratory 3PT unders were 260–120 with +8.25% actual-price ROI; that exploratory result did not replicate and must not override the independent test.

This document summarizes the immutable GitHub Actions artifact; it does not modify the preregistration or model.
