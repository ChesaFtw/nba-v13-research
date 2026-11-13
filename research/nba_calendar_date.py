"""NBA calendar-date gate for timestamped research rows."""
from datetime import datetime
from zoneinfo import ZoneInfo

def game_date_eastern(tipoff_at):
    dt=datetime.fromisoformat(tipoff_at.replace("Z","+00:00"))
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError("timezone-aware tipoff required")
    return dt.astimezone(ZoneInfo("America/New_York")).date().isoformat()
