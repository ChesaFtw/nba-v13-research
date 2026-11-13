"""Point-in-time odds/roster source audit. No network calls or betting actions."""
from collections import Counter
from datetime import datetime, timezone

def _dt(value):
    if not isinstance(value, str):
        raise ValueError("missing timestamp")
    d=datetime.fromisoformat(value.replace("Z","+00:00"))
    if d.tzinfo is None:
        raise ValueError("timestamp lacks timezone")
    return d.astimezone(timezone.utc)

def audit_quotes(rows):
    """Check observable lead-time concentration and provenance.

    Rows require observed_at_utc, tipoff_at_utc, game_id, source.
    A vendor timestamp is NOT equivalent to an independently observed time.
    """
    rows=list(rows)
    if not rows:
        raise ValueError("no quotes")
    leads=Counter()
    late=0
    for r in rows:
        if not r.get("game_id") or not r.get("source"):
            raise ValueError("quote identity/provenance missing")
        lead=(_dt(r["tipoff_at_utc"])-_dt(r["observed_at_utc"])).total_seconds()
        if lead <= 0:
            late+=1
        leads[int(round(lead/60))]+=1
    most_common,count=leads.most_common(1)[0]
    share=count/len(rows)
    return {"rows":len(rows),"non_pretip_rows":late,
            "most_common_lead_minutes":most_common,
            "most_common_lead_share":round(share,6),
            "lead_time_concentration_flag":len(rows)>=30 and share>=.95,
            "usable_for_pretip_claims":late==0 and not (len(rows)>=30 and share>=.95),
            "note":"A concentration flag is a provenance review trigger, not proof of fabrication."}
