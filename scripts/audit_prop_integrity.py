"""Audit chronological player features and detect target leakage/duplicates."""
import argparse,json
from pathlib import Path
from collections import Counter

def audit(path,season):
    p=Path(path)
    rows=[json.loads(x) for x in p.open() if x.strip()]
    if len(rows)<10000:raise RuntimeError(f"Too few features: {len(rows)}")
    keys=[(r["player_id"],r["game_id"]) for r in rows]
    duplicate=len(keys)-len(set(keys))
    dates=sorted({r["game_date"] for r in rows})
    invalid_history=sum(r["history_games"]<5 for r in rows)
    invalid_numeric=0
    for r in rows:
        for m in ("points","rebounds","assists","threes","pra"):
            for n in (5,10,20):
                if not isinstance(r.get(f"{m}_avg_{n}"),(int,float)):invalid_numeric+=1
            if not isinstance(r.get(f"{m}_actual"),(int,float)):invalid_numeric+=1
    # Arithmetic consistency is useful but is not a proof of no target leakage.
    inconsistent=sum(abs(r["pra_actual"]-(r["points_actual"]+r["rebounds_actual"]+r["assists_actual"]))>1e-6 for r in rows)
    report={"season":season,"rows":len(rows),"unique_dates":len(dates),"first_date":dates[0],"last_date":dates[-1],"duplicate_player_game":duplicate,"invalid_history":invalid_history,"invalid_numeric":invalid_numeric,"inconsistent_pra":inconsistent,"passed":not any((duplicate,invalid_history,invalid_numeric,inconsistent)),"limitation":"Structural audit only; chronological rolling-feature construction must also be code-reviewed"}
    return report

def main():
    p=argparse.ArgumentParser();p.add_argument("--season",type=int,required=True);p.add_argument("--input",required=True);a=p.parse_args()
    report=audit(a.input,a.season)
    out=Path("research_results/integrity");out.mkdir(parents=True,exist_ok=True)
    (out/f"integrity_{a.season}.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2),flush=True)
    if not report["passed"]:raise SystemExit(1)
if __name__=="__main__":main()
