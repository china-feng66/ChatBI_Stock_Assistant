from pathlib import Path
from uuid import uuid4

import pytest


@pytest.fixture
def tmp_path():
    """Workspace-local isolated temp root without recursively deleting files."""

    directory = Path(__file__).with_name(".runtime_tmp") / uuid4().hex
    directory.mkdir(parents=True, exist_ok=False)
    yield directory
