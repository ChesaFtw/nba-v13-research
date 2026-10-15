from datetime import datetime, timedelta, timezone
from research.opening_day_gate import assess_pick

def row():
    now=datetime.now(timezone.utc)
    return dict(game_id='g',market='3pm',selection='player over',line=2.5,price_american=-110,quote_at_utc=(now-timedelta(minutes=5)).isoformat(),tipoff_at_utc=(now+timedelta(hours=1)).isoformat(),injury_checked_at_utc=(now-timedelta(minutes=10)).isoformat(),model_version='v13',model_validated=True,source='timestamped sportsbook quote')

def test_valid(): assert assess_pick(row())['eligible']
def test_no_line():
    r=row();del r['line'];assert not assess_pick(r)['eligible']
def test_no_validation():
    r=row();r['model_validated']=False;assert 'model_not_validated' in assess_pick(r)['reasons']
def test_future_quote():
    r=row();r['quote_at_utc']=r['tipoff_at_utc'];assert 'quote_not_pretip' in assess_pick(r)['reasons']
def test_bad_timezone():
    r=row();r['quote_at_utc']='2026-10-20T12:00:00';assert 'invalid_market_data' in assess_pick(r)['reasons']
def test_bad_odds():
    r=row();r['price_american']=-50;assert 'invalid_american_odds' in assess_pick(r)['reasons']
