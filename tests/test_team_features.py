from v13.team_features import team_features
from v13.prediction_ledger import append_prediction
import pytest

def g(day,gid,hs,vs):
    return {"id":gid,"date":day,"home_team":{"id":1},"visitor_team":{"id":2},"home_team_score":hs,"visitor_team_score":vs,"status":"Final"}

def test_team_features_do_not_use_current_game():
    rows=team_features([g("2025-10-01",1,100,90),g("2025-10-02",2,200,50)])
    assert rows[0]["home_points_for_avg"] is None
    assert rows[1]["home_points_for_avg"]==100

def test_same_day_no_leakage():
    rows=team_features([g("2025-10-01",1,100,90),g("2025-10-01",2,200,50)])
    assert all(r["home_points_for_avg"] is None for r in rows)

def test_prediction_ledger_append_only(tmp_path):
    p=tmp_path/"ledger.jsonl"
    row={"event_id":"123","market":"points","selection":"over","model_version":"v13","predicted_at_utc":"2026-10-14T06:00:00Z"}
    append_prediction(p,row)
    append_prediction(p,row)
    assert len(p.read_text().splitlines())==2
    with pytest.raises(ValueError): append_prediction(p,{**row,"outcome":"won"})
