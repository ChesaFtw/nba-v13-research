# NBA V13 player props — research plan, Rounds 13–20

Written in response to the "Player Prop Predictive Research Master Plan". Nothing here is a measured result. Every number marked *prior* is a judgement, not data.

## 0. Starting point (measured)
- **R10:** blowout unders −2.12% (4,764 bets); consistency −2.97%. **R11:** 0 of 11 matchup features passed.
- **R12:** price outliers −6.23% (1,450 bets); line outliers −3.41% (14,983 bets). **The official injury-PDF probe retrieved no usable reports.**
- **Implication:** at open, the books are hard to beat with public box-score features, and betting against the odd book out lost money. The remaining structural opening is *information timing*. That needs prospective capture, which can only start now.

## 1. Feature registry (`research/R13_FEATURE_REGISTRY.csv`, all 310 items)

| Class | Meaning | Count |
|---|---|---|
| A | Box scores + game index we already hold, pregame-safe | 139 |
| A~ | Box-score proxy only (the proxy is stated) | 19 |
| B | `/nba/v2/stats/advanced` per-game tracking/hustle field. **Historical coverage unverified until the R13 inventory** | 59 |
| C | Play-by-play (`/nba/v1/plays`). Availability unverified; heavy engineering | 15 |
| D | Prospective capture only (injury status, restrictions, lineups announced) | 9 |
| E | Needs gated game odds (2025-26, then prospective) | 6 |
| X | Not responsibly testable | 63 |

**What the API actually offers.** `/nba/v2/stats/advanced` (per player-game, with a `period` field) lists:
- rebound chances (off/def/total), box outs, touches, passes, secondary and free-throw assists;
- contested/uncontested FGA, defended-at-rim, deflections, switches_on, speed/distance;
- matchup_* aggregates, usage, pace, shot-mix percentages.

It does **not** list potential assists, drives, time of possession, catch-and-shoot or pull-up splits, play types, defender distance, or defender identity. Season-level tracking aggregates (`/team_season_averages`) are not point-in-time, so using them historically would leak.

## 2. Thirty hypotheses, ranked
Ranking = mechanism strength × data class × chance the price misses it × cost. The prior is our honest guess that it adds information beyond the opening price.

| # | Hypothesis (mechanism) | Class | Market | Round | Prior |
|---|---|---|---|---|---|
| 1 | Vacated minutes (rotation teammate missed previous game) redistribute minutes/usage | A | PRA, pts, reb, ast | 13 | low–med |
| 2 | Minutes distribution model beats EWMA on unseen games (prediction, not edge) | A | — | 13 | high (prediction) |
| 3 | Minutes × per-minute rate NB probability adds information beyond price | A | all | 13 | low |
| 4 | Return from ≥ 3 missed games suppresses minutes (restriction) | A | PRA | 13 | low–med |
| 5 | Rebound chances per minute × opponent missed shots → rebound opportunity model | B | reb | 14 | low–med |
| 6 | Starting C/PF absent previous game → rebound redistribution to remaining bigs | A | reb | 14 | low–med |
| 7 | Rebound conversion (reb / chances) regresses; chances are the stable skill | B | reb | 14 | low |
| 8 | Opponent OREB strength cuts defensive rebound chances | A | reb | 14 | low |
| 9 | Teammate box-out volume suppresses a big's rebounds | B | reb | 14 | low |
| 10 | 3PA per minute (EWMA) is stable, 3P% is noise: lines overreact to recent makes | A | threes | 15 | low–med |
| 11 | Uncontested-FGA share shift → three-point volume change | B | threes | 15 | low |
| 12 | Primary creator absent previous game → fewer assisted threes for spot-up shooters | A/B | threes | 15 | low |
| 13 | Touches/passes per minute are stable assist opportunity; assists/pass regresses | B | ast | 16 | low–med |
| 14 | Teammate shooting availability (best shooters absent) lowers assists | A | ast | 16 | low |
| 15 | Secondary playmaker promoted when the primary is absent | A | ast | 16 | low–med |
| 16 | Usage redistribution with an absent star (usage_percentage history) | A/B | pts, PRA | 17 | low |
| 17 | FTA rate × opponent foul rate | A | pts | 17 | low |
| 18 | Shot-mix (paint share) × opponent rim defense (defended_at_rim) | B | pts | 17 | low |
| 19 | Schedule density × years since draft × recent minutes load | A | PRA | 18 | low |
| 20 | Structural break (role change) detected earlier than the line adjusts | A | PRA | 18 | low–med |
| 21 | Hot hand: recent production beyond opportunity/shot quality (expect null) | A/B | all | 18 | very low |
| 22 | Foul susceptibility × opponent FTA rate → minutes risk | A | PRA | 18 | low |
| 23 | Opponent pace (estimated possessions) × player rates | A/B | PRA | 19 | low |
| 24 | Opponent 3PA rate → long rebounds for guards/wings | A | reb | 14 | low |
| 25 | New-team integration after a trade (minutes ramp) | A | PRA | 18 | low |
| 26 | Rookie minutes progression through the season | A | PRA | 18 | low |
| 27 | Lineup continuity (top-5 overlap) and efficiency | A~ | pts | 19 | very low |
| 28 | Spread-implied competitiveness × player minutes distribution (H1 failed at the bet level) | E | PRA | 19 | very low |
| 29 | Combo vs component pricing coherence (consistency failed in R10) | A | combos | 19 | very low |
| 30 | Steals/blocks: deflections and defended_at_rim opportunity, only if priced two-sided | B | stl/blk | 19 | unknown (inventory) |

## 3. Data acquisition plan
1. **R13 inventory (this round, about 30 requests):**
   - `/nba/v2/stats/advanced` for 2023-24, 2024-25 and 2025-26 (3 dates each): per-field non-null share, especially rebound_chances, touches, passes, contested_fga, box_outs;
   - `/nba/v1/advanced` for comparison;
   - `/nba/v1/plays` (both parameter styles), `/nba/v1/lineups`, and the `/nba/v1/player_injuries` snapshot schema;
   - prop types and two-sided coverage in the saved 2025-26 opening props.
2. **If B fields cover all three seasons at ≥ 80%:** collect full per-game advanced v2, keeping period 0 rows only, then build R14–R17 B features with the same strictly-earlier-day rule. If they cover only 2025-26, B features are prospective/one-season only, and the plan says so.
3. **Plays (C):** collect only if R14+ needs it *and* the inventory shows substitutions/coordinates for past seasons. 2025-26 first.
4. **Lineups:** 2025-26 onward, prior games only (the endpoint returns data once a game begins, so the current game's lineup is never a pregame input).
5. **Prospective capture from opening night 2026-27 (the most valuable step; it cannot be done later):**
   - odds logger v2, which needs `main`;
   - **injury-status snapshots** every 15–30 min, stamped with our clock (`scripts/capture_injuries.py`, workflow `r13-injury-snapshots.yml`). `status_changes()` derives first-seen times. These are the only injury timestamps a backtest may use.
6. **Not acquired:** scheme/coverage labels, wingspan/reach, birth dates, defender-pair matchups, stats.nba.com tracking scrape (cloud-IP blocking and terms are unverified, so not planned without a deliberate decision).

## 4. Round 13
See `PREREG_R13_OPPORTUNITY.md`, `v13/minutes_model.py`, `scripts/r13_opportunity.py`, `scripts/r13_data_inventory.py`, `scripts/capture_injuries.py` and tests `tests/test_round13.py`.

## 5. Roadmap with gates
| Round | Work | Gate to continue |
|---|---|---|
| 13 | Inventory; minutes distribution; opportunity information test; start injury snapshots | Part A PASS → minutes model becomes the baseline. Part B result is screening only |
| 14 | Rebounds: opportunity (missed shots × chances) then share; positional absence | Information PASS vs **market + minutes baseline** |
| 15 | Threes: attempts first, makes regress to long-run %; creator absence | Same |
| 16 | Assists: touches/passes opportunity, teammate shooting availability | Same |
| 17 | Points: usage, FTA, shot mix vs rim defense | Same |
| 18 | Fatigue, form, structural breaks, trades/rookies | Same; hot-hand expected null |
| 19 | Combine only survivors; calibrated ensemble; cross-market correlations (pts/threes/PRA are not independent) | Calibration (reliability curves), log loss vs market on 2026-27 data |
| 20 | Prospective shadow qualification on synchronized logger prices at the user's daily scan time; CLV; frozen staking | ≥ 300 shadow bets, positive CLV, cluster CI > 0 before any real stake |

**Test-budget rule.** 2025-26 is our only validated price season. R11–R13 have screened it 25 times. From R14 on, the 2025-26 window is used for *screening* only. Every "PASS" is confirmed only on 2026-27 data captured prospectively, with the feature frozen first. Prediction-only tests (box scores, no prices) can use all seasons freely.

## 6. Signals we cannot test responsibly with current data
- **Coverage scheme:** drop, switch rate (beyond the switches_on count), zone, blitz, trap, double-team, help, closeouts.
- **Physical / matchup:** wingspan, standing reach, true age; expected primary defender and defender-pair history; post-up/isolation/pick-and-roll play types per game.
- **Tracking we don't have:** potential assists, time of possession, drives, catch-and-shoot vs pull-up per game, defender distance per shot, xFG%.
- **News and intent:** coach comments and intent, chemistry, minutes restrictions historically, official injury designations and timestamps historically (the probe failed), starting-lineup announcements historically.
- **Too little data or unverified pricing:** arena effects (samples too small), alternate/milestone lines and first-quarter props (pricing semantics unverified).

## 7. Market priority
1. **Rebounds:** opportunity-driven, the richest tracking support (chances, box outs), less public attention.
2. **Assists:** teammate-dependent; touches/passes give an opportunity measure.
3. **PRA / combos:** where a minutes model has the most leverage (all three stats scale with minutes).
4. **Threes:** attempts are predictable, makes are noisy. Heavy binomial variance limits any edge.
5. **Points:** the most efficient and most heavily bet market. Lowest prior.
6. **Steals/blocks/doubles:** only if the inventory shows two-sided prices with enough volume. High variance and typically heavy juice.
