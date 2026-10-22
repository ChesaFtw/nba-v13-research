# NBA V13 — factual implementation status (updated 2026-10-21)

V13 remains a **research platform, NOT a live trained/calibrated wagering model**.
Never infer a profitable bet from a single external score projection or a raw historical hit rate.

## Now working and verified
- NBA regular-season official-source-oriented archival: ESPN full game summaries,
  complete play-by-play, final player/team box scores, coach rotations via recorded
  substitutions, points/rebounds/assists/turnovers, individual +/−, field-goal/3PA/FTA
  volume, shot events, period scores, paint points, transition proxy, estimated
  team possessions, four-factor features, pace and offensive efficiency.
- NBA CDN produces HTTP 403 on hosted GitHub Actions; a tested ESPN public
  scoreboard/summary source works (HTTP 200), while BALLDONTLIE is available
  for existing historical collection.
- GitHub Actions `.github/workflows/live-game-archive.yml` runs scheduled collection
  twice daily and can be run manually. It collects recently completed regular-season
  games across the NBA, validates final scores against PBP, archives compressed
  full raw summaries per game, writes normalized features and commits them into
  `research_data/live_nba/YYYY-MM-DD/`. No silent success for missing finals.
- Initial successful Actions run saved **8** finalized games across Oct 20–21,
  including Milwaukee-Washington game `401909835`. This count will change when
  more games settle; it is not a season-total claim.
- Chronological prior-game feature builder `scripts/build_prior_only_game_features.py`
  captures rolling team/playmaking/scoring/rebounding/pace, prior known roster,
  schedule/rest features. Actual future-game starters/players are NOT included
  as pregame input; same-date results cannot leak.
- Safety testing in repo `pytest`, GitHub Actions checks passed on 2026-10-21
  before additional tests.
- Detailed loss audit: `research/BUCKS_WIZARDS_POSTMORTEM_2026-10-21.md`.
- New stringent approved-bet gate `research/pregame_approval_gate.py` requires
  out-of-sample market-specific calibration, timestamp discipline, team churn,
  injury/lineup checks and a lower confidence bound on edge. Until calibration
  passes this gate should fail closed.

## Still missing; DO NOT claim accomplished
1. **Actual deployable forecast model** that is calibrated and compared with
   standard market baseline on real historical lines using walk-forward holdouts.
2. **Multi-season complete feature store** with consistent ESPN or BDL fields;
   historical scripts exist but there is no certified joined training corpus.
3. **Accurate lineup-stint possession ratings** and optical defender matchups.
   Player plus-minus and substitution logs are archived, but possession-labeled
   simultaneous lineups have not been fully validated.
4. **Precise possession clock duration, defensive matchup height/wingspan and
   shot-quality tracking.** ESPN summaries do not guarantee these, so external
   verified/licensed sources and quality tests are required.
5. **Timestamped full sportsbook market history** and injured-player minutes
   projections as of each prediction, necessary for credible true-odds EV/CLV.
6. **Documented retraining and promotion policy:** gather seasons, use rolling
   feature histories, fit/calibrate with no leakage, test out of sample, compare
   to baseline, only promote when statistically reliable. Eight games cannot
   validate the accuracy of a totals, spread or prop model.

## Privacy and betting ledger
Real customer stakes are not pushed into this **public** research repository.
Private official bets are maintained separately from public basketball history.

## Research parameters
- Provisional −110 research odds when true odds unavailable, clearly labeled.
- Never present assumed odds as executable Caesars lines.
- No betting against the New York Knicks.
- Betting abstention when model or evidence is missing.
