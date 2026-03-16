from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from switchyard.bench.cli import app


def test_run_synthetic_cli_writes_artifact(tmp_path: Path) -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "--request-count",
            "2",
            "--output-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    artifact_path = Path(result.stdout.strip())
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert payload["request_count"] == 2
    assert payload["policy"] == "balanced"
    assert payload["summary"]["success_count"] == 2
