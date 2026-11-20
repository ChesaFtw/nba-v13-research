import pytest
from research.appearance_data_gate import validate_appearance_universe, join_injury_status

def candidates():
    return [
        {"game_id":1,"player_id":10,"tipoff_at_utc":"2026-10-20T23:00:00Z",
         "roster_observed_at_utc":"2026-10-20T20:00:00Z","appeared":True,"source":"roster"},
        {"game_id":1,"player_id":11,"tipoff_at_utc":"2026-10-20T23:00:00Z",
         "roster_observed_at_utc":"2026-10-20T20:00:00Z","appeared":False,"source":"roster"}]

def test_includes_dnp():
    assert validate_appearance_universe(candidates())["did_not_appear"] == 1
def test_no_dnp_rejected():
    with pytest.raises(ValueError, match="zero DNP"):
        validate_appearance_universe(candidates()[:1])
def test_post_tip_rejected():
    rows=candidates();rows[0]["roster_observed_at_utc"]=rows[0]["tipoff_at_utc"]
    with pytest.raises(ValueError, match="before tipoff"):
        validate_appearance_universe(rows)
def test_duplicate_rejected():
    rows=candidates();rows.append(dict(rows[0]))
    with pytest.raises(ValueError, match="duplicate"):
        validate_appearance_universe(rows)
def test_injury_future_not_leaked():
    rows=candidates()
    snaps=[{"player_id":10,"captured_at_utc":"2026-10-20T21:00:00Z","status":"Questionable"},
           {"player_id":10,"captured_at_utc":"2026-10-20T23:01:00Z","status":"Out"}]
    out=join_injury_status(rows,snaps)
    assert out[0]["injury_status_pretip"]=="Questionable"
    assert out[1]["injury_status_pretip"] is None
def test_missing_injury_is_unknown_not_healthy():
    assert join_injury_status(candidates(),[])[0]["injury_status_pretip"] is None
