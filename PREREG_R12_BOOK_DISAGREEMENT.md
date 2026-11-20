# Pre-registration — R12 sportsbook disagreement (2025-26 gated opening props)

Frozen before any result is seen. Code: `scripts/book_disagreement.py` (constants in `FROZEN`).

## Question
When one sportsbook's opening prop disagrees with the other books, does betting against that book beat its own price?
No projection of ours is used: the other books are the model.

## Data
- 2025-26 opening player props, provenance-gated (`odds-gate-1`). The script fails closed on ungated rows. 2023-24 and 2024-25 stay quarantined.
- Markets: points, rebounds, assists, threes, PRA. Over/under only; both prices required.
- One quote per book per player-game-market: that book's earliest gated row.
- Outcome from the regular-season box score. No box score = DNP, void. Pushes are excluded from ROI.

## Hypotheses (both frozen; each must pass on its own)
**H12a — price outlier at the same line.**
- Comparison books: those quoting the same line, opened within 60 minutes of the outlier book's quote; at least 2 required. At least 3 books must quote the player-market.
- Fair P(over) = mean of the comparison books' power-de-vigged P(over).
- Bet the side whose expected value at the outlier book's actual price is ≥ +3%.
- One bet per player-game-market (largest EV; ties broken by fixed vendor priority).

**H12b — line outlier.**
- The book's line differs by ≥ 1.0 from the median line of the other books that opened within 60 minutes (at least 2 required).
- Bet the side that book made easier (over a lower line, under a higher line), only if that side is priced −150 or better.
- One bet per player-game-market (largest gap).

## PASS (all, per hypothesis)
- ≥ 300 decided bets;
- ROI > 0 at actual prices;
- z ≥ 2.5 for hit rate vs the outlier book's own de-vigged probability, under both multiplicative and power de-vig;
- game-cluster bootstrap 95% CI lower bound on ROI > 0.

## Descriptive only (cannot rescue a FAIL)
By market, side, outlier vendor, and posting hour (US Central). Also excess vs the consensus probability: we expect it near zero if the consensus is right.

## What a PASS does and doesn't mean
- A PASS means the opening outlier prices were beatable. It doesn't show that a once-a-day bettor could get them. We only see opening quotes, and an outlier may be corrected within minutes.
- A PASS licenses exactly one next step: a frozen 2026-27 shadow test run from a once-daily scheduled scan, graded on prices actually observed by the live logger at scan time.
- A FAIL retires both rules. No changes to thresholds, windows, markets or vendors after seeing results.
