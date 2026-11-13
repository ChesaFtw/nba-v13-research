# Injury first-seen provenance audit — 2026-10-19 19:00 CT

Research-only, no model activation. Audit source: `scripts/capture_injuries.py` SHA `6c19739a29e9ec5906bde38dfa42de507500e221`.

Findings: collector timestamps `captured_at_utc` before the HTTP pagination requests and lacks a `finished_at_utc`; its `status_changes()` accepts arbitrary snapshot lists without a completeness gate. A truncated snapshot can produce false `not_listed` status events. Audit-branch snapshot and odds workflows are absent from `main`, so their scheduled execution cannot be presumed; release listing contains zero releases as of this inspection.

A standalone local fail-closed audit utility and 12 synthetic tests were executed (12 passed). The utility requires completed pagination, consistent counts/hashes/unique player IDs, and a verified capture completion timestamp no later than forecast cutoff. This is NOT a test of production collector integration, nor proof of actual injury capture. Current manifests fail the completion-time gate.

Next: add end-of-capture timestamp to the collector behind an explicit review gate; validate pagination before computing status changes; verify scheduling and archival strategy without merging to main. Historical accuracy impact unmeasured due to missing verified point-in-time injury snapshots.
