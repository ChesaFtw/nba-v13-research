"""Apply the historical-odds eligibility gate to one raw odds file. Nothing is deleted.

Writes <out>/eligible.jsonl (stamped, usable with --validated), <out>/excluded.jsonl (stamped
with the exclusion reason, kept for forensics) and <out>/summary.json.

python -m scripts.quarantine_odds --source bdl_opening_player_props --games games_2025.jsonl \
    --odds opening_props_2025.jsonl --out research_data/quarantine/props_2025
"""
import argparse, json
from pathlib import Path

from v13.provenance import SOURCES, quarantine


def read(p):
    return [json.loads(x) for x in Path(p).open() if x.strip()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, choices=sorted(SOURCES))
    p.add_argument("--games", action="append", required=True, help="one or more game index files")
    p.add_argument("--odds", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    games = [g for path in a.games for g in read(path)]
    eligible, excluded, summary = quarantine(read(a.odds), games, a.source)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "eligible.jsonl").write_text("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in eligible))
    (out / "excluded.jsonl").write_text("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in excluded))
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
