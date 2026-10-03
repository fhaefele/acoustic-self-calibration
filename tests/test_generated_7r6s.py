from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from acoustic_self_calibration.stratified.generated_7r6s import (
    ALL_MINOR_SPECS,
    EVENT_COUNT,
    GENERIC_COMPLEX_SOLUTION_COUNT,
    PRIMARY_MINOR_SPECS,
    RECEIVER_COUNT,
    TEMPLATE_SHA256,
)


def test_generated_7r6s_template_contract() -> None:
    assert RECEIVER_COUNT == 7
    assert EVENT_COUNT == 6
    assert GENERIC_COMPLEX_SOLUTION_COUNT == 5
    assert len(PRIMARY_MINOR_SPECS) == 6
    assert len(ALL_MINOR_SPECS) == 75
    assert len(TEMPLATE_SHA256) == 64
    primary = {tuple(map(tuple, item)) for item in PRIMARY_MINOR_SPECS}
    all_specs = {tuple(map(tuple, item)) for item in ALL_MINOR_SPECS}
    assert len(primary) == len(PRIMARY_MINOR_SPECS)
    assert primary.issubset(all_specs)


def test_generator_reproduces_checked_in_template(tmp_path: Path) -> None:
    output = tmp_path / "generated_7r6s.py"
    subprocess.run(
        [
            sys.executable,
            "tools/generate_7r6s_template.py",
            "--output",
            str(output),
        ],
        check=True,
    )
    checked_in = Path("src/acoustic_self_calibration/stratified/generated_7r6s.py").read_text(
        encoding="utf-8"
    )
    assert output.read_text(encoding="utf-8") == checked_in
