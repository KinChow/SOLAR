"""Bounded subprocess execution with durable logs."""

import asyncio
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Sequence


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str
    duration_ms: float


class CommandRunner:
    async def run(
        self,
        command: Sequence[str],
        *,
        log_path: Path,
        timeout_seconds: float,
    ) -> CommandResult:
        started = monotonic()
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout_seconds,
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.communicate()
            self._write_log(
                log_path,
                command,
                stdout="",
                stderr="",
                trailer=f"[timeout]\nstage exceeded {timeout_seconds:g} second timeout",
            )
            raise TimeoutError(f"stage exceeded {timeout_seconds:g} second timeout")
        except asyncio.CancelledError:
            process.kill()
            await process.communicate()
            raise
        stdout = stdout_bytes.decode(errors="replace")
        stderr = stderr_bytes.decode(errors="replace")
        duration_ms = (monotonic() - started) * 1_000
        self._write_log(log_path, command, stdout=stdout, stderr=stderr)
        return CommandResult(process.returncode, stdout, stderr, duration_ms)

    @staticmethod
    def _write_log(
        log_path: Path,
        command: Sequence[str],
        *,
        stdout: str,
        stderr: str,
        trailer: str = "",
    ) -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        suffix = f"\n{trailer}" if trailer else ""
        log_path.write_text(
            f"$ {' '.join(command)}\n\n[stdout]\n{stdout}\n[stderr]\n{stderr}{suffix}"
        )
