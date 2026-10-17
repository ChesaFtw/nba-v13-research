# NBA V13 — Three-track verification audit (2026-10-15)

Research only. No merge, staking, production activation, or betting.

## Track 1 — historical prediction integrity
- Repository contains `scripts/walk_forward_prop_mae.py` (expanding chronological Ridge vs last-10, projection MAE only) and `scripts/holdout_spreads_totals.py` (chronological opening-line evaluation).
- `CHALLENGER_GATE_2026-10-14.md` records a successful 2025 pilot run 38584634800, but does not certify historical artifact persistence or independent pricing coverage.
- `research/EXPERIMENT_A_2024_THREES_UNDERS_RESULT.md`: 2024–25 three-point unders 232/394 (58.88%), recorded-price ROI -3.55%, replication FAIL. Three-point overs 524/878, ROI -3.62%.
- `research/EXPERIMENT_REBOUNDS_2024_RESULT.md`: 482/918 rebounds, recorded-price ROI -7.99%, replication FAIL.
- Both historical prop experiments have explicit vendor coverage caveats; do not reuse their 2024–25 quotes as validated independent market evidence.
- GitHub releases listing returned zero releases at inspection; existing durable historical data beyond Actions artifacts remains unverified. Do not claim data were downloaded.

## Track 2 — injury and price collectors
- `scripts/capture_injuries.py` uses BALLDONTLIE `/nba/v1/player_injuries`, capture timestamps, SHA256 hashes, pagination and status transitions.
- `scripts/log_live_odds.py` logs vendor timestamps, capture times, odds, provenance and raw responses.
- Workflows `r13-injury-snapshots.yml` and `live-odds-logger.yml` exist on research branch and write weekly releases if run.
- GitHub scheduled workflows run from the default branch, so research-branch schedule declarations alone are not evidence that they run. Do not merge these workflows to main under research-only scope.
- API key availability, actual API response, run frequency, and data retention remain unverified. No secrets should be printed.

## Track 3 — independent model comparisons
- The existing walk-forward evaluator compares Ridge with a last-10 baseline on prior-date features, not actual betting results.
- Research-only `research/minutes_cluster_bootstrap.py` and tests were committed on this branch. CI run 38761033424 passed; these are software tests, not empirical model improvement.
- Before any challenger promotion, require matching games, identical timestamp cutoffs, strictly pre-tip features, chronological training, independent test folds, uncertainty intervals, and baseline-relative MAE / calibration / hit-rate by market.
- Never claim 65% hit rate or positive ROI without untouched independent validation.

## Blockers / next executable gates
1. Identify and retrieve accessible artifacts or archived data manifests; verify row counts, season type, timestamps and SHA256.
2. Verify collector workflow dispatch / runtime and API authentication without revealing credentials; record nonzero coverage and errors.
3. Feed matched held-out minutes predictions into cluster audit; run spread/total and prop comparisons on identical chronological folds.
4. Retain -110 solely as provisional research pricing where actual price is absent; never invent market thresholds or claim executable odds.
