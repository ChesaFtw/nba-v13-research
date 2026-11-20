# Pre-registration H1: blowout-driven minutes loss in starter props (2025–26)

**Committed before any H1 statistic is computed.** No prior project analysis has conditioned starter props on the pre-game spread.

## Hypothesis and mechanism
In lopsided games, high-minute starters sit late, which fattens the left tail of their minutes. If opening prop prices are built mainly from per-game averages, then **UNDERS** on those starters in high-|spread| games will beat the de-vigged opening price.

## Data (provenance-gated only)
- Opening player props: the `bdl_opening_player_props` gate.
- Opening game spreads: the `bdl_opening_game_odds` gate.
- Both 2025–26 regular season. Box scores are played-minutes rows only, so DNP props (voided by books) are excluded.

## Frozen definitions (`scripts/h1_blowout_minutes.py`, `FROZEN`)
- **High-minute starter:** average ≥ 30.0 minutes over his last 10 played games before the game, with at least 5 such games.
- **Team expected margin:** taken from the gated opening spread.
  - Blowout group: |margin| ≥ 9.5.
  - Competitive control: |margin| ≤ 4.5.
  - Everything in between is excluded.
- **Markets:** points, rebounds, assists, threes. Over/under only.
- **Line selection:** fixed vendor priority, then the earliest gated row. One line per player-game-market.
- **Bet:** the UNDER at its recorded opening price. Pushes = 0 P&L.

## Frozen pass rule (blowout group, all markets pooled)
**PASS only if all hold:**
- ≥ 300 decided bets;
- ROI at actual prices > 0;
- excess ≥ 2.0 pp and z ≥ 2.5 under both multiplicative and power de-vig;
- game-cluster bootstrap ROI 95% CI lower bound > 0;
- the mechanism holds (starters' minutes relative to their prior average are lower in the blowout group than in the competitive group);
- the blowout group's power-de-vig excess is higher than the competitive control's.

**Anything else is a FAIL.** Per-market results are descriptive only and may not rescue a FAIL.

## What a PASS would and would not mean
2025–26 is the only documented season. A PASS would show an **information** effect in opening prices: the price missed something knowable before the game. It would not show an executable edge. It would license only a pre-registered 2026–27 shadow test, with the same frozen definitions, graded at logger-captured prices and on closing-line value. It would not license staking.
