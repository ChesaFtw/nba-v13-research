# Pre-registration — R14 three-pointers and rebounds

Frozen before any R14 result is seen.
- **Code:** `v13/r14_features.py`, `v13/r14_models.py`, `scripts/r14_research.py` (constants `FOLDS`, `TESTS`, `TRACKING_TESTS`, `GATE`, `REB_PHYS`, `REB_OTHER`, `SHADOW_THRESHOLD`) and `scripts/collect_tracking.py`.
- **Workflow:** `r14-research.yml`.

## Status of the R13 threes PASS going in
- **The PASS:** R13 `threes|model_edge` (minutes model × EWMA 3PM-per-minute → negative binomial, vs the power-de-vigged opening price) improved test-window log loss by 0.001371 and passed BH, the sign check and the CI rule.
- **Its weakness:** it was one of 25 tests on the same 2025-26 window. There is **no untouched historical test period left** for 2025-26 prices.
- **Therefore:** R14 *explains* it on that window (descriptive), *replicates* it on 2024-25 (unverified prices, exploratory), and *freezes* it for prospective 2026-27 confirmation.

## Data
- **Box scores** 2023-24, 2024-25, 2025-26 (played rows) and the game index.
- **Prices:** gated 2025-26 opening props (fails closed). The 2024-25 opening props (quarantined) are used only for the exploratory replication.
- **Tracking:** `/nba/v2/stats/advanced` full seasons collected by `collect_tracking.py`. Period-0 rows only. Used **only as lagged prior-game features** and **only if the gate passes**.

## Features (strictly earlier days; see module docstrings)
**Threes:**
- 3PA per minute (EWMA, half-life 10);
- 3P% shrunk to the league-to-date rate with K = 150 attempts;
- hot_gap (last-5 3P% − shrunk, ≥ 10 attempts);
- attempt_trend (last-5 − last-20 3PA/min);
- vacated_ast.

**Rebounds:**
- expected defensive and offensive rebound opportunities (averages of opponent and own misses, last 10);
- DREB and OREB share per opportunity-minute (EWMA, shrunk to the position mean with K = 5 games);
- opp_oreb_rate, vacated_reb, opp_fg3a_rate, b2b, 3-in-4;
- **physical terms:** opponent minutes-weighted size (centered), × expected misses, × the player's height, × projected minutes.

  The player's own height is constant, so his share history absorbs it. Only the opponent-size part of "height advantage" varies by game, and that is what the GLM estimates. The "height advantage × misses × minutes" interaction is the model's structure: rebounds ∝ minutes × opportunities × share × exp(physical terms).

**Tracking (if gated in):**
- rebound chances per minute (EWMA);
- conversion = REB / chances, shrunk to the league rate with K = 30.

## Part A — prediction only (no prices), walk-forward
- **Folds:** train 2023-24 → test 2024-25; train 2023-24 + 2024-25 → test 2025-26. The minutes model is refit per fold.
- **Comparisons:**

  | Comparison | Model | Against |
  |---|---|---|
  | Threes | V1 = NB attempts (μ = predicted min × 3PA/min) × Binomial(shrunk 3P%) | V0 = the R13 model |
  | Rebounds | GLM: Poisson IRLS with offset log(predicted min/48 × (opps_d × share_d + opps_o × share_o)), ridge 1, NB dispersion | B0 = R13 minutes × EWMA rebound rate |
  | Physical ablation | GLM with the four physical terms | GLM without them |
  | Tracking (if gated) | chances model | GLM |

- **Metric:** mean NB / compound log score of the actual stat, game-cluster bootstrap CI. MAE is reported too.
- **PASS:** CI lower bound > 0 **in both folds**.

## Part B — information beyond the de-vigged 2025-26 opening price (screening)
- **Harness:** same as R13, generalized to an incremental baseline. Logistic on logit(price) + base features, then + the tested feature. Train before 2026-02-01, test after, game-cluster bootstrap.
- **Tests:**

  | # | Market | Feature | Baseline adds |
  |---|---|---|---|
  | 1 | threes | edge_v1 | — |
  | 2 | threes | edge_v1 | edge_v0 |
  | 3 | threes | attempt_trend | — |
  | 4 | threes | hot_gap | — |
  | 5 | threes | vacated_ast | — |
  | 6 | rebounds | edge_struct | — |
  | 7 | rebounds | edge_struct | edge_b0 |
  | 8 | rebounds | vacated_reb | — |

  Plus, only if the tracking gate passes: rebounds edge_track; rebounds edge_track beyond edge_struct.
- **PASS:** Δ < 0, 95% CI upper bound < 0, BH q = 0.10 across all R14 tests run, and sign stable across the two training halves. Brier scores are reported too.
- **Cumulative screening on this window:** R11 11 + R12 2 + R13 12 + R14 8–10. A PASS here is a candidate for prospective confirmation, nothing more.

## Tracking coverage gate (frozen)
Tracking tests run only if, for **each** of 2023-24, 2024-25 and 2025-26:
- ≥ 90% of box-score player-games with ≥ 10 minutes have a non-null period-0 `rebound_chances_total`;
- the period-sum consistency for that field is ≥ 95% wherever period rows exist.

Otherwise tracking is reported as untestable for this design. It is **never** backfilled with season aggregates.

## Descriptive only (cannot pass or fail anything)
- **R13 threes diagnosis**, by: line, model direction, predicted minutes, projected attempts, month, vendor. Edge deciles with hit rate and opening-price ROI toward the model.
- **2024-25 replication** (exploratory, models trained on 2023-24 only).
- **Vacated-minutes audit:** same-night returns, trade phantoms, never-reappeared.
- **Tracking manifests:** coverage by field and month, period consistency, duplicates, HTTP status / 429s / back-off, elapsed time.

## Frozen 2026-27 shadow recipe (R13 threes)
- **Recipe:** the minutes model and threes dispersion are refit on 2023-24..2025-26. The recalibration (intercept, market logit, edge z) is fit on all 2025-26 threes rows. Both are written to the artifact before any 2026-27 game.
- **Decision rule (shadow only, flat 1 unit):** bet toward the recalibrated probability when |p_recal − p_devig| ≥ 0.03, at the price the logger observed at the daily scan time.
- **Shadow success (judged at ≥ 300 shadow bets):**
  - closing-line value > 0 with game-cluster CI lower bound > 0;
  - prospective log-loss Δ < 0 with CI upper bound < 0.

  ROI at captured prices is reported but is not the criterion at this sample size.
- **No real stake** before shadow success. Then flat 1% (tier A) only.
