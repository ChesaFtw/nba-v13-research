"""Guard: pytest's default patterns (test_*.py, *_test.py; test_* functions) must not match
production modules, or CI collects helpers as tests (Round 11 CI failure)."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_no_collectable_test_functions_outside_tests():
    bad = []
    for d in ("scripts", "v13"):
        for p in (ROOT / d).rglob("*.py"):
            if re.match(r"(test_.*|.*_test)\.py$", p.name):
                for i, line in enumerate(p.read_text().splitlines(), 1):
                    if re.match(r"\s*def test", line) or re.match(r"\s*class Test", line):
                        bad.append(f"{p.relative_to(ROOT)}:{i}")
    assert not bad, bad


def test_pytest_ini_restricts_collection_to_tests():
    text = (ROOT / "pytest.ini").read_text()
    assert re.search(r"^testpaths\s*=\s*tests\s*$", text, re.M)
