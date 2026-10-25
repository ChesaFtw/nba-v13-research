# NBA V13 22:00 Central — injury collector/event integration audit (2026-10-19)

Scope: **research-only**, audit/claude-2026-10-14. No main merge, model activation, staking change, or wagers.

## Inputs inspected
- Existing audit branch `scripts/capture_injuries.py` SHA `6c19739a29e9ec5906bde38dfa42de507500e221` (unchanged).
- Local prior 20:00 collector candidate and 21:00 event extractor from timestamped research packages.
- GitHub Actions CI run 39344735090: success for existing branch, **not** the new local candidate.
- Readiness run 39254879992: ten unexpired artifact records; release listing returned zero releases.
- Official BALLDONTLIE NBA docs show `/v1/player_injuries`; collector uses `/nba/v1/player_injuries`. Authenticated endpoint/entitlement check not performed.

## New end-to-end local tests
Built 7 synthetic tests exercising actual candidate collector -> manifest -> event extractor:
1. Questionable -> Out, timestamped at **capture completion**.
2. Multi-page response hash/JSONL roundtrip.
3. Truncated pagination breaks adjacency.
4. Empty snapshot and relisting never imply medical clearance.
5. Snapshot finishing after cutoff excluded.
6. Different source URLs must not be compared as same status feed.
7. Missing source metadata must fail closed.

**Baseline: 5 passed, 2 failed.** The failures demonstrated that the 21:00 extractor accepted a manifest with no `source` and emitted a cross-source status change for the same player ID.

Research-only correction: require a nonempty HTTPS `source` field, compare source URLs before status transitions, break the chain on mismatch, and count source mismatches. Existing standalone fixtures updated to include provenance.

**Retest: 44 passed, 0 failed** (7 integration + 23 extractor + 14 collector tests). `py_compile` passed. No real NBA captures, authenticated API calls, live workflow tests, remote CI, or demonstrated prediction accuracy gain.

## Decision
Do not activate. Keep frozen minutes model. Candidate code/tests remain local until branch-only integration can be committed and tested remotely; this note does not deploy code. Source URL changes must wait for authenticated smoke tests. GitHub schedule triggers on the default branch, so audit-only cron definitions do not establish operational collection.
