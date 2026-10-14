# NBA V13 — foundation status

This repository is an **early, unvalidated research scaffold**, not a trained or backtested betting model.

## Implemented
- Secret-based BALLDONTLIE games endpoint ingestion with pagination
- A scheduled GitHub Actions data inventory and manual trigger
- Transparent implied-probability and expected-return math
- Unit tests in CI

## Not implemented
- Historical point-in-time datasets, odds history, injuries, player props
- Trained or calibrated predictive models, independent challengers
- Nested walk-forward backtesting, bankroll tracking, graded picks
- Caesars Kansas sportsbook prices or 15-minute-before-tip dynamic scheduling
- Automated alerts or persistent data warehouse

## Security
Never commit API keys. Add BALLDONTLIE_API_KEY at Settings > Secrets and variables > Actions.
The workflow will fail until that secret is configured. Workflow cron is UTC and is only an
approximation of the intended Central Time window; daylight saving and per-tip scans remain TODO.

## Betting research
Default -110 is an assumed research price only, not a live executable quote.
Do not interpret any output as a validated edge.
