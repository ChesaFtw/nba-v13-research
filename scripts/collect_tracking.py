"""Collect /nba/v2/stats/advanced for one season and audit it before anything uses it.

Writes  tracking_<season>.jsonl   period-0 (full-game) rows only, every field kept
        manifest_<season>.json    the coverage audit:
          requests, HTTP status counts, 429s and total back-off seconds, elapsed time, pages, rows by period
          per-field non-null share of period-0 rows, overall and by month
          period_consistency: for count fields, share of player-games whose period rows sum to the
            period-0 value (within 1), where period rows exist
          duplicates: repeated (game, player, period) keys
Usability is decided by the R14 coverage gate, not by "the endpoint responded".
Timing caveat recorded in the manifest: values are post-game and may have been revised by the NBA
after the fact; they are only used as LAGGED (prior-game) features.
"""
import argparse, json, os, time
from collections import Counter, defaultdict
from pathlib import Path

URL = "https://api.balldontlie.io/nba/v2/stats/advanced"
CHECK = ("rebound_chances_total", "touches", "passes", "box_outs", "contested_fga")


class Auditor:
    """Streaming audit: period-0 rows go straight to `sink`; only small aggregates stay in memory."""

    def __init__(self, sink=None):
        self.sink = sink
        self.by_period = Counter(); self.keys = Counter(); self.p0 = 0
        self.nonnull = Counter(); self.fields = set()
        self.month_n = Counter(); self.month_nonnull = defaultdict(Counter)
        self.sums = defaultdict(lambda: defaultdict(float)); self.seen = defaultdict(set)
        self.full_vals = {}

    def add(self, r):
        gid = str((r.get("game") or {}).get("id", r.get("game_id")))
        pid = str((r.get("player") or {}).get("id", r.get("player_id")))
        per = r.get("period")
        self.by_period[str(per)] += 1
        self.keys[(gid, pid, str(per))] += 1
        if per in (0, None):
            self.p0 += 1
            m = str((r.get("game") or {}).get("date", ""))[:7]
            self.month_n[m] += 1
            for k, v in r.items():
                if isinstance(v, dict):
                    continue
                self.fields.add(k)
                if v is not None:
                    self.nonnull[k] += 1
                    if k in CHECK:
                        self.month_nonnull[m][k] += 1
            self.full_vals[(gid, pid)] = {f: r.get(f) for f in CHECK}
            if self.sink is not None:
                self.sink(r)
        else:
            for f in CHECK:
                if r.get(f) is not None:
                    self.sums[(gid, pid)][f] += float(r[f]); self.seen[(gid, pid)].add(f)

    def finish(self):
        cons = {}
        for f in CHECK:
            ok = tot = 0
            for k, vals in self.full_vals.items():
                if f in self.seen.get(k, ()) and vals.get(f) is not None:
                    tot += 1; ok += abs(self.sums[k][f] - float(vals[f])) <= 1.0
            cons[f] = round(ok / tot, 4) if tot else None
        n = self.p0 or 1
        return {"rows_by_period": dict(self.by_period), "period0_rows": self.p0,
                "duplicate_keys": sum(1 for v in self.keys.values() if v > 1),
                "field_nonnull_period0": {k: round(self.nonnull[k] / n, 4) for k in sorted(self.fields)},
                "key_fields_by_month": {m: {f: round(self.month_nonnull[m][f] / c, 4) for f in CHECK}
                                        for m, c in sorted(self.month_n.items())},
                "period_consistency": cons}


def audit(all_rows):
    """Convenience wrapper (tests): -> (period-0 rows, audit dict)."""
    full = []
    a = Auditor(full.append)
    for r in all_rows:
        a.add(r)
    return full, a.finish()


def main():
    import requests
    p = argparse.ArgumentParser()
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-pages", type=int, default=4000)
    p.add_argument("--sleep", type=float, default=0.25)
    a = p.parse_args()
    key = os.environ.get("BALLDONTLIE_API_KEY")
    if not key:
        raise SystemExit("BALLDONTLIE_API_KEY missing")
    s = requests.Session()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    fh = (out / f"tracking_{a.season}.jsonl").open("w")
    aud = Auditor(lambda r: fh.write(json.dumps(r, separators=(",", ":")) + "\n"))
    raw, status, backoff, cursor, pages, seen_cursors = 0, Counter(), 0.0, None, 0, set()
    t0 = time.time()
    for _ in range(a.max_pages):
        params = {"seasons[]": a.season, "per_page": 100}
        if cursor is not None:
            params["cursor"] = cursor
        for attempt in range(8):
            try:
                r = s.get(URL, headers={"Authorization": key}, params=params, timeout=60)
            except requests.RequestException:
                status["network_error"] += 1; time.sleep(10 * (attempt + 1)); continue
            status[str(r.status_code)] += 1
            if r.status_code in (429, 500, 502, 503, 504):
                try:
                    wait = min(120.0, max(1.0, float(r.headers.get("Retry-After", ""))))
                except ValueError:
                    wait = min(120.0, 10.0 * (attempt + 1))
                backoff += wait; time.sleep(wait); continue
            break
        r.raise_for_status()
        body = r.json(); pages += 1
        for row in body.get("data", []):
            aud.add(row); raw += 1
        cursor = (body.get("meta") or {}).get("next_cursor")
        if cursor is None:
            break
        if str(cursor) in seen_cursors:
            raise SystemExit("repeated cursor")
        seen_cursors.add(str(cursor))
        if pages % 100 == 0:
            print(f"season {a.season}: pages={pages} rows={raw} elapsed={time.time() - t0:.0f}s", flush=True)
        time.sleep(a.sleep)
    fh.close()
    manifest = {"season": a.season, "source": URL, "complete": cursor is None, "pages": pages, "raw_rows": raw,
                "requests_by_status": dict(status), "backoff_seconds": round(backoff, 1),
                "elapsed_seconds": round(time.time() - t0, 1), **aud.finish(),
                "timing_caveat": "Post-game values, possibly revised after the fact; use only as lagged prior-game features."}
    (out / f"manifest_{a.season}.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k not in ("field_nonnull_period0", "key_fields_by_month")}, indent=2))


if __name__ == "__main__":
    main()
