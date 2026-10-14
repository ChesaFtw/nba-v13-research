"""Keep only played player stats from a verified regular-season game set."""

def filter_regular(games, stats):
    ids={g['id'] for g in games if not g.get('postseason')}
    return [r for r in stats if r.get('game',{}).get('id') in ids and r.get('min') not in (None,'','0','0:00')]
