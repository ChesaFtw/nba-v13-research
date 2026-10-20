import pytest
from research.quote_provenance_audit import audit_quotes

def row(tip="2026-10-20T23:00:00Z",seen="2026-10-20T20:00:00Z"):
    return {"game_id":1,"source":"recorded_api","tipoff_at_utc":tip,"observed_at_utc":seen}

def test_uniform_three_hour_lead_flagged():
    report=audit_quotes([row() for _ in range(40)])
    assert report["lead_time_concentration_flag"]
    assert not report["usable_for_pretip_claims"]

def test_late_quote_not_usable():
    report=audit_quotes([row(seen="2026-10-20T23:01:00Z")])
    assert report["non_pretip_rows"]==1
    assert not report["usable_for_pretip_claims"]

def test_mixed_pretip_not_flagged():
    rows=[row(seen=f"2026-10-20T{hour:02d}:00:00Z") for hour in (19,20,21,22)]
    assert audit_quotes(rows)["usable_for_pretip_claims"]

def test_missing_provenance_rejected():
    r=row();r["source"]=None
    with pytest.raises(ValueError,match="provenance"):
        audit_quotes([r])
