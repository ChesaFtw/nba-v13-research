import pytest

from v13.minutes_forecast_audit import evaluate_minutes_forecasts


def test_challenger_reduces_error():
    rows = [
        {"actual": 30, "baseline": 25, "challenger": 29,
         "predicted_at_utc": "2026-10-20T16:00:00Z",
         "tipoff_at_utc": "2026-10-20T23:00:00Z"},
        {"actual": 24, "baseline": 29, "challenger": 25,
         "predicted_at_utc": "2026-10-20T16:00:00Z",
         "tipoff_at_utc": "2026-10-20T23:00:00Z"},
    ]
    result = evaluate_minutes_forecasts(rows)
    assert result["n"] == 2
    assert result["baseline_mae"] == 5
    assert result["challenger_mae"] == 1
    assert result["mae_improvement"] == 4
    assert result["challenger_better"]


@pytest.mark.parametrize("prediction_time", [
    "2026-10-20T23:00:00Z", "2026-10-20T23:01:00Z",
])
def test_rejects_post_tipoff_predictions(prediction_time):
    with pytest.raises(ValueError, match="before tipoff"):
        evaluate_minutes_forecasts([{
            "actual": 30, "baseline": 29, "challenger": 30,
            "predicted_at_utc": prediction_time,
            "tipoff_at_utc": "2026-10-20T23:00:00Z",
        }])


def test_rejects_missing_timezone():
    with pytest.raises(ValueError, match="timezone"):
        evaluate_minutes_forecasts([{
            "actual": 30, "baseline": 29, "challenger": 30,
            "predicted_at_utc": "2026-10-20T16:00:00",
            "tipoff_at_utc": "2026-10-20T23:00:00Z",
        }])


def test_rejects_nonfinite_values():
    with pytest.raises(ValueError, match="finite"):
        evaluate_minutes_forecasts([{
            "actual": 30, "baseline": float("nan"), "challenger": 30,
            "predicted_at_utc": "2026-10-20T16:00:00Z",
            "tipoff_at_utc": "2026-10-20T23:00:00Z",
        }])
