"""Check training/holdout selection and frozen benchmark results."""
from scripts.benchmark_prop_models import evaluate
def test_date_holdout_and_model_selection():
    rows=[]
    for d in range(100):
        for pid in range(20):
            row={"game_date":f"2025-01-{d+1:03d}","player_id":pid}
            for market in ("points","rebounds","assists","threes","pra"):
                row[f"{market}_actual"]=10.
                for window,value in ((5,10.),(10,20.),(20,30.)):
                    row[f"{market}_avg_{window}"]=value
            rows.append(row)
    result=evaluate(rows,2025)
    assert result["training_rows"]==1600
    assert result["holdout_rows"]==400
    assert result["models"]["points"]["selected_on_train"]=="last5"
    assert result["models"]["points"]["holdout_selected_mae"]==0
