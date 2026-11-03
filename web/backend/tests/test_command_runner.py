import sys
from pathlib import Path
from time import monotonic

import pytest

from solar_web.workers.command_runner import CommandRunner


@pytest.mark.asyncio
async def test_kills_stage_process_when_timeout_expires(tmp_path: Path) -> None:
    started = monotonic()
    log_path = tmp_path / "timeout.log"

    with pytest.raises(TimeoutError, match="0.05 second timeout"):
        await CommandRunner().run(
            [sys.executable, "-c", "import time; time.sleep(5)"],
            log_path=log_path,
            timeout_seconds=0.05,
        )

    assert monotonic() - started < 1
    assert "[timeout]" in log_path.read_text()
