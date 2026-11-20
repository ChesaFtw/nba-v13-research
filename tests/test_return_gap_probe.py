from research.return_gap_probe import build_rows, summarize

def fixture():
    games=[]; players=[]
    for i in range(15):
        game={"id":i+1,"date":f"2026-01-{i+1:02d}","status":"Final","postseason":False,
              "home_team":{"id":1},"visitor_team":{"id":2}}
        games.append(game)
        if i < 10 or i == 13:
            players.append({"game":game,"team":{"id":1},"player":{"id":50},
                            "min":"10" if i==13 else "20"})
    return games,players

def test_three_missed_team_games():
    games,players=fixture()
    rows=build_rows(games,players)
    assert len(rows)==1
    assert rows[0]["gap_team_games"]==3
    assert rows[0]["signed_error_last10"]==-10

def test_future_games_do_not_change_past_row():
    games,players=fixture()
    before=build_rows(games,players)
    extra={"id":16,"date":"2026-01-16","status":"Final","postseason":False,
           "home_team":{"id":1},"visitor_team":{"id":2}}
    games.append(extra)
    assert build_rows(games,players)==before

def test_zero_minutes_not_counted_as_appearance():
    games,players=fixture()
    players[0]["min"]="0"
    assert build_rows(games,players)==[]

def test_new_team_history_not_shared():
    games,players=fixture()
    players[-1]["team"]={"id":2}
    assert build_rows(games,players)==[]

def test_empty_cohort():
    assert summarize([])=={"n":0}
