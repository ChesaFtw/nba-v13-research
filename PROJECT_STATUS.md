# NBA V13 — implementation status

Early, **unvalidated** research scaffold. No trained model, no betting picks.

## Implemented
- BALLDONTLIE games ingestion using GitHub Actions secret, pagination, explicit preseason and regular-season filters
- Diagnostic scan for current and next Central-time date plus historical regular-season control (2024-10-22)
- Unit tests for market math and request parameters; GitHub Actions CI
- Scheduled scan and manual workflow trigger

## Still required
- Confirm diagnostic workflow outputs nonzero historical games and preseason games when scheduled
- Persist datasets and build chronological, leakage-safe feature pipeline
- Build and calibrate predictive models; nested walk-forward out-of-sample validation
- Injuries, player props, market snapshots, bankroll tracker, alerts, 15-minute-before-tip scans

## Notes
The NBA games API excludes preseason by default, so explicit season_type=preseason is essential.
Dates are computed using America/Chicago, with two dates queried to cover UTC boundaries.
No API keys are committed. GitHub Actions must have BALLDONTLIE_API_KEY configured.
Default -110 is a hypothetical research assumption, not an executable sportsbook price.
