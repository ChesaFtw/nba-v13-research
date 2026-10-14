"""Bounded NBA GOAT API research collection, with honest availability manifests."""
import argparse,json,os,time
from datetime import datetime,timezone
from pathlib import Path
import requests

ENDPOINTS={
 "advanced_v1":("https://api.balldontlie.io/nba/v1/stats/advanced","seasons[]"),
 "advanced_v2":("https://api.balldontlie.io/nba/v2/stats/advanced","seasons[]"),
 "opening_odds":("https://api.balldontlie.io/nba/v2/odds/opening",None),
 "injuries_snapshot":("https://api.balldontlie.io/v1/player_injuries",None),
}
def collect(kind,season,out,max_pages=600):
    key=os.environ.get("BALLDONTLIE_API_KEY")
    if not key: raise RuntimeError("BALLDONTLIE_API_KEY missing")
    url,season_key=ENDPOINTS[kind]
    client=requests.Session();cursor=None;seen=set();count=0;pages=0
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    target=out/f"{kind}_{season if season is not None else 'snapshot'}.jsonl"
    with target.open("w") as f:
        for page in range(max_pages):
            params={"per_page":100}
            if season_key:params[season_key]=season
            if cursor is not None:params["cursor"]=cursor
            for attempt in range(8):
                try:
                    response=client.get(url,headers={"Authorization":key},params=params,timeout=40)
                    if response.status_code in (429,500,502,503,504):
                        if attempt==7:response.raise_for_status()
                        raw=response.headers.get("Retry-After","")
                        try:wait=min(120,max(1,float(raw)))
                        except ValueError:wait=min(10*(attempt+1),120)
                        print(f"{kind} retry HTTP {response.status_code} after {wait}s",flush=True)
                        time.sleep(wait);continue
                    response.raise_for_status()
                    payload=response.json()
                    break
                except requests.RequestException:
                    if attempt==7:raise
                    time.sleep(min(10*(attempt+1),60))
            rows=payload.get("data",[])
            if not isinstance(rows,list):raise RuntimeError("Unexpected API payload")
            for row in rows:f.write(json.dumps(row,separators=(",",":"))+"\n")
            count+=len(rows);pages+=1
            nxt=(payload.get("meta") or {}).get("next_cursor")
            if pages%25==0 or nxt is None:print(f"{kind} season={season} pages={pages} rows={count}",flush=True)
            if nxt is None:cursor=None;break
            if str(nxt) in seen:raise RuntimeError("Repeated API cursor")
            seen.add(str(nxt));cursor=nxt
            time.sleep(0.4)
    complete=cursor is None
    report={"dataset":kind,"season":season,"rows":count,"pages":pages,"complete":complete,"retrieved_at_utc":datetime.now(timezone.utc).isoformat(),"source":url,"note":"Snapshot only; historical availability varies by endpoint. Do not assume point-in-time injury or prop history."}
    (out/f"{kind}_{season if season is not None else 'snapshot'}_manifest.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2),flush=True)
    if not complete:raise RuntimeError("Page cap reached; data incomplete")
    if not count:raise RuntimeError("No records returned; dataset not available")
def main():
    p=argparse.ArgumentParser();p.add_argument("--kind",choices=ENDPOINTS,required=True);p.add_argument("--season",type=int);p.add_argument("--max-pages",type=int,default=600);p.add_argument("--out",default="research_data/extra_sources")
    a=p.parse_args();collect(a.kind,a.season,a.out,a.max_pages)
if __name__=="__main__":main()
