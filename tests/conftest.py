from pathlib import Path

import pytest


@pytest.fixture
def tmp_path():
    """Workspace-local replacement for an inaccessible Windows temp root."""

    directory = Path(__file__).with_name(".runtime_tmp")
    directory.mkdir(exist_ok=True)
    trace_file = directory / "trace.jsonl"
    trace_file.unlink(missing_ok=True)
    try:
        yield directory
    finally:
        trace_file.unlink(missing_ok=True)
