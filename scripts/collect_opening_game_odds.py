"""Save raw BALLDONTLIE opening spread/total/moneyline records (with prices) for one season.

Keeps every vendor row unchanged so later scripts can apply pre-tip and vendor rules.
python -m scripts.collect_opening_game_odds --games research_data/game_history/games_2024.jsonl \
    --out research_data/game_odds/opening_2024.jsonl
"""
import argparse, json, os
from collections import Counter
from pathlib import Path

from scripts.backtest_opening_spreads_totals import fetch, load


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--games", required=True); p.add_argument("--out", required=True)
    a = p.parse_args()
    key = os.environ.get("BALLDONTLIE_API_KEY")
    if not key:
        raise RuntimeError("BALLDONTLIE_API_KEY not set")
    games = [g for g in load(a.games) if g.get("postseason") is False]
    rows = fetch([g["id"] for g in games], key)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with_prices = sum(r.get("spread_home_odds") is not None and r.get("total_over_odds") is not None for r in rows)
    manifest = {"games": len(games), "rows": len(rows), "games_with_rows": len({str(r.get('game_id')) for r in rows}),
                "rows_with_spread_and_total_prices": with_prices,
                "vendors": Counter(str(r.get("vendor")) for r in rows)}
    (out.parent / (out.stem + "_manifest.json")).write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
