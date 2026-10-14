"""Collect immutable, timestamped NBA official injury-report snapshots.

Only report statuses with a verifiable publication time earlier than the chosen
pregame cutoff. Never substitute current injury data for historical reports.
"""
import argparse,json,re,time
from datetime import date,datetime,timedelta,timezone
from pathlib import Path
import requests
from io import BytesIO

URL="https://ak-static.cms.nba.com/referee/injury/Injury-Report_{day}_{clock}.pdf"
CLOCKS=("05_15AM","06_00PM","05_00PM","01_00PM","11_00AM","08_00PM","12_00PM","09_00AM","07_00PM","03_00PM")
def collect(start,end,out):
    from pypdf import PdfReader
    target=Path(out);target.parent.mkdir(parents=True,exist_ok=True)
    session=requests.Session()
    total=0;found=0;missing=0
    with target.open("w") as fh:
        day=date.fromisoformat(start);stop=date.fromisoformat(end)
        while day<=stop:
            for clock in CLOCKS:
                url=URL.format(day=day.isoformat(),clock=clock)
                try:
                    r=session.get(url,timeout=25)
                    if r.status_code in (403,404):missing+=1;continue
                    r.raise_for_status()
                    if not r.content.startswith(b"%PDF"):continue
                    reader=PdfReader(BytesIO(r.content))
                    pages=[p.extract_text() or "" for p in reader.pages]
                    if not any("Injury Report:" in p for p in pages):continue
                    # Time is encoded in source URL, in ET; retain raw PDF text
                    # for later auditable game/player-level extraction.
                    record={"report_date":day.isoformat(),"clock_et":clock,
                        "source_url":url,"pages":pages,
                        "sha256":__import__("hashlib").sha256(r.content).hexdigest(),
                        "report_kind":"historical_official_snapshot"}
                    fh.write(json.dumps(record)+"\n");found+=1
                except (requests.RequestException,ValueError,Exception) as exc:
                    # Do not fabricate availability when source cannot be parsed.
                    print(f"Snapshot unavailable {url}: {type(exc).__name__}",flush=True)
                time.sleep(.015)
            total+=1;day+=timedelta(days=1)
    manifest={"dates_scanned":total,"snapshots_found":found,"missing_urls":missing,
        "data_type":"timestamped raw reports only; no model performance claimed",
        "source":"NBA official injury report PDFs"}
    target.with_suffix(".manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps(manifest))
    if found==0:raise RuntimeError("No historical official NBA injury report snapshots accessible")

def main():
    p=argparse.ArgumentParser();p.add_argument("--start",required=True);p.add_argument("--end",required=True);p.add_argument("--out",required=True)
    a=p.parse_args();collect(a.start,a.end,a.out)
if __name__=="__main__":main()
