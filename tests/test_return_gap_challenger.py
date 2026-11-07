from research.return_gap_challenger import fit_gap_correction

def test_empty_training_fails():
    try:
        fit_gap_correction([], '2026-01-15')
    except ValueError:
        return
    assert False
