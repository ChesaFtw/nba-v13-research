# Pre-registration — R15 market discovery (points, assists, steals, blocks, combos)

Frozen before any R15 result is seen.
- **Code:** `v13/r15_features.py` and `scripts/r15_research.py` (constants `FOLDS`, `TESTED_MARKETS`, `COVS`, `ABSENCE_RULE`, `FIELD_GATE`, `MIN_TWO_SIDED`, `tests_for()`).
- **Workflow:** `r15-research.yml`.
- **Three-pointers are not retested.** They are frozen and confirmed only prospectively, by the ledger.

## Market feasibility (from the data source's documentation; re-measured in the run)
- **Prop types the source lists:** points, rebounds, assists, threes, steals, blocks, points_rebounds, points_assists, rebounds_assists, points_rebounds_assists, double_double, triple_double, and 1st-quarter and first-3-minute variants.
- **Not offered:** turnovers, free throws made, field goals made, three-point attempts, first-half props. These can't be priced, so they aren't modeled.
- **Double/triple-doubles** are milestone (one-sided) odds, which can't be de-vigged.
- **1Q and first-3-minute outcomes** need play-by-play, which box scores don't have.
- **A market is modeled** only if the gated 2025-26 file has ≥ 2,000 two-sided rows for it.

## Absence v2 (replaces the R13/R14 proxy, which was wrong 39% of the time)
- **Definition:** a rotation teammate (last-10 average ≥ 15 min when last seen) who has missed the team's last **k or more consecutive** games (at most 15).
- **Returning teammate:** played the team's previous game after missing ≥ 3 straight.
- **Choosing k:** chosen on **2023-24 box scores only** (no prices). Take the smallest k in {1, 2, 3} for which the share of flagged teammates who nevertheless played that night is ≤ 25%. If none qualifies, the absence features are dropped.
- **Why 25%:** every absence ends with one return game that no box-score rule can foresee. The leftover misclassification biases toward finding *no* effect.
- **The real fix** is the prospective injury snapshots.

## Tracking field gate
A tracking-based model is fitted only if each field it uses is non-null on ≥ 90% of box-score player-games with ≥ 10 minutes, in each of 2023-24, 2024-25 and 2025-26:

| Model | Field |
|---|---|
| assists | passes |
| steals | deflections |
| blocks | defended_at_rim_fga |

The audit is reported by field × season × minutes role (10–15, 15–28, 28+) × month.

## Shared opportunity foundation (strictly earlier days)
- **Minutes:** the R13 minutes model, refit per fold.
- **Rates:** per-minute EWMAs (half-life 10 games): FGA2, 3PA, FTA, AST, STL, BLK, TOV, PTS, REB, usage volume, passes, touches, deflections, rim contests.
- **Conversions:** career-to-date, shrunk to the league-to-date rate (FG2 K = 100, FG3 K = 150, FT K = 50, AST/pass K = 200, STL/deflection K = 30, BLK/rim contest K = 40).
- **Context:** pace ratio, opponent points / assists / turnovers per possession (relative to league), opponent 2PA share, teammate eFG, usage trend, fouls per 36, rest flags, absence v2 and returning teammates.

## Models
- **Base (R13-style):** projected minutes × per-minute rate, negative binomial.
- **Opportunity GLMs** (Poisson with offset = opportunity × conversion; NB dispersion; ridge 1):

  | Market | Offset | Covariates |
  |---|---|---|
  | points | min × (2PA·2·FG2% + 3PA·3·FG3% + FTA·FT%) | pace, opponent defence, vac_usage_k, returning usage, usage trend, b2b, 3-in-4 |
  | assists | min × passes × AST/pass | teammate eFG, opponent assists allowed, pace, vac_ast_k, returning assists, b2b |
  | steals | min × deflections × STL/deflection | opponent turnover rate, pace, fouls/36, b2b |
  | blocks | min × rim contests × BLK/contest | opponent 2PA share, pace, fouls/36, b2b |

- **Assists comparison model:** the same covariates with offset = min × recent assists per minute. This tests "opportunity beats recent results".
- **Combos (PR, PA, RA):** a joint model (NB on the sum, with dispersion fitted on the sum, so shared minutes and correlation are in the spread) vs an independent convolution of the component NBs.

## Part A — prediction (no prices), walk-forward 2023→2024 and 2023+24→2025
Mean log-score gain with a game-cluster CI. PASS = CI lower bound > 0 in both folds. Comparisons:
- opportunity vs base (points, assists, steals, blocks);
- passes vs recent-assists offset;
- joint vs independent (PR, PA, RA).

## Part B — information beyond the de-vigged 2025-26 opening price (screening)
**Tests:**

| Market | Tests |
|---|---|
| points | edge_opp; edge_opp beyond edge_base; returning_usage; vac_usage* |
| assists | edge_opp; edge_opp beyond edge_base; returning_ast; vac_ast* |
| steals | edge_base; edge_opp; edge_opp beyond edge_base |
| blocks | edge_base; edge_opp; edge_opp beyond edge_base |
| combos | edge_combo for PR, PA, RA |

\* Only if absence v2 is retained.

**Not retested (already failed):** points/assists/PRA base edges (R13), rebounds (R14).

**PASS:** Δ log loss < 0, 95% CI upper bound < 0, BH q = 0.10 across all R15 tests run, and sign stable across training halves.

**Reported for every test (descriptive):**
- results split by rows where the challenger leans over vs leans under. Both directions are always reported; neither can be selected after the fact;
- a 10-bin calibration table and ECE;
- Brier scores.

## Promotion rule (frozen)
A Part B PASS goes to prospective shadow tracking only if:
- the same edge's 2024-25 replication (exploratory prices, models trained on 2023-24 only) has Δ < 0;
- every tracking field it uses passed the field gate.

**Cumulative screening on the 2025-26 window:** R11 11 + R12 2 + R13 12 + R14 8 + R15 up to 17. No PASS here is confirmation.

## Real money
Not authorized by any result in this round. The ledger records every eligible prop in shadow mode. Shadow success for a market:
- ≥ 300 qualified bets with closing prices;
- mean CLV > 0 with game-cluster CI lower bound > 0;
- prospective log-loss CI upper bound < 0.
