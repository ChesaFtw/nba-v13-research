"""Offline audit of cached NBA season manifests; never triggers API requests.

Usage: python scripts/audit_cached_seasons.py path/to/manifests/*.json
Exit nonzero if any manifest is incomplete, truncated, duplicated, or inconsistent.
"""
import argparse
import json
from pathlib import Path


def audit(paths):
    seen = set()
    reports = []
    for path in paths:
        record = json.loads(Path(path).read_text())
        season = record.get("season")
        if not isinstance(season, int) or isinstance(season, bool):
            raise ValueError(f"{path}: missing integer season")
        if season in seen:
            raise ValueError(f"{path}: duplicate season {season}")
        seen.add(season)
        if record.get("complete") is not True or record.get("truncated") is not False:
            raise ValueError(f"{path}: collection not explicitly complete and untruncated")
        reports.append({"season": season, "manifest": str(path), "status": "complete"})
    if not reports:
        raise ValueError("No manifests supplied")
    return sorted(reports, key=lambda x: x["season"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifests", nargs="+")
    args = parser.parse_args()
    print(json.dumps(audit(args.manifests), indent=2))


if __name__ == "__main__":
    main()
