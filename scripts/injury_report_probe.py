"""R12 probe: can we collect the official NBA injury reports for 2025-26 with trustworthy timestamps?

This is a feasibility probe, not an experiment. It answers four questions before any collector is built:
  1. Does the official index page list the season's report PDFs, and how many?
  2. What filename/time format do they use (hourly "_05PM" or 15-minute "_05_30PM")?
  3. Do the PDFs download, and what do their Last-Modified headers say vs the filename time?
  4. Can the text be extracted well enough to parse (player, team, game, status)?
It saves raw text for a spread of sample reports so the parser is written against the real layout.

Publication time comes from the filename (US Eastern). That is the timestamp the H3 design relies on.
"""
import argparse, json, re
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

INDEX_URLS = ("https://official.nba.com/nba-injury-report-2025-26-season/",)
PDF_BASE = "https://ak-static.cms.nba.com/referee/injury/"
LINK_RE = re.compile(r"https?://[^\"'\s>]*Injury-Report_\d{4}-\d{2}-\d{2}_[0-9_]+[AP]M\.pdf", re.I)
NAME_RE = re.compile(r"Injury-Report_(\d{4}-\d{2}-\d{2})_(\d{1,2})(?:_(\d{2}))?([AP]M)\.pdf$", re.I)
ET = ZoneInfo("America/New_York")
HEADERS = {"User-Agent": "Mozilla/5.0 (research probe; contact via repository)"}


def extract_links(html):
    return sorted(set(LINK_RE.findall(html or "")))


def report_time(url):
    """Filename time -> (eastern_iso, utc_iso, format) or None."""
    m = NAME_RE.search(url)
    if not m:
        return None
    day, hh, mm, ampm = m.group(1), int(m.group(2)), int(m.group(3) or 0), m.group(4).upper()
    hh = hh % 12 + (12 if ampm == "PM" else 0)
    et = datetime.fromisoformat(f"{day}T{hh:02d}:{mm:02d}:00").replace(tzinfo=ET)
    return {"eastern": et.isoformat(), "utc": et.astimezone(timezone.utc).isoformat(),
            "format": "quarter_hour" if m.group(3) else "hourly"}


def candidate_urls(day):
    """Fallback if the index page is unavailable: both known filename styles for one date."""
    out = []
    for h in range(1, 13):
        for ampm in ("AM", "PM"):
            out.append(f"{PDF_BASE}Injury-Report_{day}_{h:02d}{ampm}.pdf")
            for mm in ("00", "15", "30", "45"):
                out.append(f"{PDF_BASE}Injury-Report_{day}_{h:02d}_{mm}{ampm}.pdf")
    return out


def spread_sample(items, k):
    if len(items) <= k:
        return list(items)
    step = (len(items) - 1) / (k - 1)
    return [items[round(i * step)] for i in range(k)]


def pdf_text(data):
    try:
        from io import BytesIO
        from pypdf import PdfReader
    except ImportError:
        return None, "pypdf_not_installed"
    try:
        reader = PdfReader(BytesIO(data))
        return "\n".join((p.extract_text() or "") for p in reader.pages), None
    except Exception as e:  # report, do not crash the probe
        return None, f"extract_error: {type(e).__name__}: {e}"


def text_stats(text):
    lines = [x for x in (text or "").splitlines() if x.strip()]
    return {"lines": len(lines), "status_mentions": dict((s, len(re.findall(rf"\b{s}\b", text or "")))
                                                        for s in ("Out", "Questionable", "Doubtful", "Probable", "Available")),
            "first_lines": lines[:60]}


def main():
    import requests
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--sample", type=int, default=20)
    p.add_argument("--fallback-dates", nargs="*", default=["2025-11-05", "2025-12-22", "2026-01-20", "2026-02-25", "2026-03-30"])
    a = p.parse_args()
    out = Path(a.out); (out / "text").mkdir(parents=True, exist_ok=True)
    summary = {"index": {}, "links": 0, "formats": {}, "samples": [], "fallback": {}}
    links = []
    for u in INDEX_URLS:
        try:
            r = requests.get(u, headers=HEADERS, timeout=30)
            found = extract_links(r.text)
            summary["index"][u] = {"status": r.status_code, "links": len(found)}
            links += found
        except Exception as e:
            summary["index"][u] = {"error": f"{type(e).__name__}: {e}"}
    links = sorted(set(links), key=lambda x: (report_time(x) or {}).get("utc", x))
    summary["links"] = len(links)
    times = [report_time(x) for x in links]
    summary["formats"] = {f: sum(1 for t in times if t and t["format"] == f) for f in ("hourly", "quarter_hour")}
    if times and times[0]:
        summary["first_report"], summary["last_report"] = times[0]["eastern"], times[-1]["eastern"]
    if not links:
        for day in a.fallback_dates:
            hits = []
            for u in candidate_urls(day):
                try:
                    if requests.head(u, headers=HEADERS, timeout=15).status_code == 200:
                        hits.append(u)
                except Exception:
                    pass
            summary["fallback"][day] = hits
            links += hits
    for u in spread_sample(links, a.sample):
        rec = {"url": u, "filename_time": report_time(u)}
        try:
            r = requests.get(u, headers=HEADERS, timeout=60)
            rec.update({"status": r.status_code, "bytes": len(r.content), "last_modified": r.headers.get("Last-Modified"),
                        "content_type": r.headers.get("Content-Type")})
            if r.status_code == 200:
                text, err = pdf_text(r.content)
                rec["extract_error"] = err
                if text is not None:
                    st = text_stats(text)
                    rec.update({"lines": st["lines"], "status_mentions": st["status_mentions"]})
                    name = u.rsplit("/", 1)[-1].replace(".pdf", ".txt")
                    (out / "text" / name).write_text(text)
                    rec["first_lines"] = st["first_lines"]
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"
        summary["samples"].append(rec)
    ok = [s for s in summary["samples"] if s.get("status") == 200 and s.get("lines")]
    summary["verdict"] = {"reports_listed": len(links), "samples_downloaded_and_parsed": len(ok),
                          "feasible": bool(links) and len(ok) >= max(1, len(summary["samples"]) // 2)}
    (out / "probe_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "samples"}, indent=2))
    for s in summary["samples"]:
        print(s["url"].rsplit("/", 1)[-1], s.get("status"), s.get("bytes"), s.get("last_modified"), s.get("lines"),
              s.get("status_mentions"), s.get("extract_error") or s.get("error") or "")


if __name__ == "__main__":
    main()
