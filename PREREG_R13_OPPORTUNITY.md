# Pre-registration — R13 minutes / opportunity model

Frozen before any result is seen.
- Code: `v13/minutes_model.py` (features, model, NB probabilities) and `scripts/r13_opportunity.py` (constants `TRAIN_SEASONS`, `TEST_SEASON`, `REGISTRY`, `PASS_A`; R11's `SPLIT_DATE`, `FDR_Q`).
- Workflow: `r13-opportunity.yml`.

## Data
- Regular-season box scores (played rows, min > 0) for 2023-24, 2024-25 and 2025-26, plus the game index. Box scores have no price-provenance problem, so all three seasons are usable for Part A.
- Part B uses only 2025-26 opening props that pass the provenance gate (`odds-gate-1`). It fails closed on ungated rows.

## Features (all from games on strictly earlier days)
**Minutes history:**
- OT-adjusted minutes for the last 3/5/10/20 games, an EWMA (half-life 5) and the 10-game SD;
- trend (last 3 − last 20).

**Schedule:**
- days of rest (capped at 5), back-to-back, 3-in-4, 4-in-6;
- log games played.

**Absence and role:**
- own team games missed since the last appearance; return flag (≥ 3 missed);
- **vacated minutes**: summed last-10 average minutes of rotation teammates (≥ 15 min) who did not play in the team's *previous* game. This is the pregame-safe absence proxy; the current game's box score is never used;
- top-5-minutes role in the previous game;
- fouls per 36.

## Part A — minutes forecasting (a prediction result, not an edge)
- **Model:** ridge regression (λ = 10, standardized features), trained on 2023-24 + 2024-25 and scored on every 2025-26 player-game with ≥ 3 prior games.
- **Distribution:** empirical residual quantiles (10/50/90) from training, by predicted-minutes bucket (< 15, 15–25, 25–32, ≥ 32).
- **Primary baseline:** EWMA minutes. Last-10 average is reported as well.
- **PASS A (all required):**
  - mean (|error model| − |error EWMA|) < 0;
  - game-cluster bootstrap 95% CI upper bound < 0;
  - 80% interval coverage between 0.77 and 0.83.

## Part B — information beyond the opening price (2025-26 gated props)
- **Projection:** stat mean = predicted minutes × EWMA per-minute rate (half-life 10 games; games with ≥ 5 min). Count distribution = negative binomial, with dispersion fitted on 2023-24 + 2024-25.
- **Features tested** (each against the R11 baseline: logistic of the over outcome on logit of the power-de-vigged opening price):

  | Market | Features |
  |---|---|
  | points | `model_edge` (logit P_model − logit P_market), `vacated_min`, `trend` |
  | rebounds | `model_edge`, `vacated_min` |
  | assists | `model_edge`, `vacated_min` |
  | threes | `model_edge` |
  | pra | `model_edge`, `vacated_min`, `return_flag` |

  That's 12 tests.
- **Split:** train before 2026-02-01, test on or after.
- **Uncertainty:** game-cluster bootstrap.
- **Multiple testing:** BH-FDR q = 0.10 across the 12 tests.
- **PASS B (per test):** test log-loss Δ < 0, **95% CI upper bound < 0**, BH-significant, and coefficient sign stable across the two training halves. The CI condition is stricter than R11.

## Multiplicity across rounds (stated up front)
The 2025-26 prices have already been screened in R11 (11 tests) and R12 (2 tests). R13 adds 12 more. So a Part B PASS is a candidate only. Confirmation requires a frozen 2026-27 prospective test on prices captured by the live logger.

## What does not count
- Descriptive slices (vacated > 0, return, b2b, trend) can't rescue a FAIL.
- No retuning of λ, half-lives, the rotation threshold, buckets or features after results are seen.
- A Part A PASS alone is not evidence of profit: the books model minutes too.
