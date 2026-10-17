from datetime import date,timedelta
from research.return_gap_incremental_gate import evaluate

def synthetic():
    rows=[]
    for i in range(90):
        d=(date(2025,12,1)+timedelta(days=i)).isoformat()
        for p in range(3):
            rows.append({"date":d,"game_id":i+1,"player_id":p,
                         "x":[i%5,p,1],"gap_team_games":3 if i%7==0 else 0,
                         "has_team_history":1,"minutes_actual":20+i%5+p,
                         "pregame_verified":True})
    return rows

def test_chronological():
    result=evaluate(synthetic(),windows=[("2026-01-15","2026-02-01")])
    assert result["2026-01-15"]["all"]["n"]==51

def test_reject_unverified():
    rows=synthetic()
    rows[0]["pregame_verified"]=False
    try:
        evaluate(rows,windows=[("2026-01-15","2026-02-01")])
    except ValueError:
        return
    assert False,"expected provenance failure"
