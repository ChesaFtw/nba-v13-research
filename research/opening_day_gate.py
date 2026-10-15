"""Research-only NBA opening-day pick eligibility gate. No wager execution."""
from datetime import datetime, timezone
from math import isfinite

REQUIRED = ('game_id','market','selection','line','price_american','quote_at_utc','tipoff_at_utc','injury_checked_at_utc','model_version','model_validated','source')

def _dt(value):
    d=datetime.fromisoformat(value.replace('Z','+00:00'))
    if d.tzinfo is None or d.utcoffset() is None:
        raise ValueError('timestamps must include UTC offset')
    return d.astimezone(timezone.utc)

def assess_pick(record):
    """Fail closed on missing provenance, future/same-tip data, unvalidated models."""
    reasons=[]
    for key in REQUIRED:
        if key not in record or record[key] is None or record[key] == '':
            reasons.append('missing_'+key)
    if reasons:
        return {'eligible':False,'reasons':reasons}
    try:
        tip=_dt(record['tipoff_at_utc'])
        quote=_dt(record['quote_at_utc'])
        injury=_dt(record['injury_checked_at_utc'])
        if quote>=tip: reasons.append('quote_not_pretip')
        if injury>=tip: reasons.append('injury_not_pretip')
        if quote>datetime.now(timezone.utc): reasons.append('quote_in_future')
        line=float(record['line']); price=int(record['price_american'])
        if not isfinite(line): reasons.append('invalid_line')
        if price == 0 or -100 < price < 100: reasons.append('invalid_american_odds')
    except (ValueError,TypeError,OverflowError):
        reasons.append('invalid_market_data')
    if record['model_validated'] is not True:
        reasons.append('model_not_validated')
    if not str(record['source']).strip(): reasons.append('missing_source')
    return {'eligible':not reasons,'reasons':reasons}
