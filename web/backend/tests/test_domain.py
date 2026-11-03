import asyncio
import json
from pathlib import Path

import pytest

from solar_web.domain.models import RunRecord, RunStatus, StageId, StageStatus


@pytest.mark.asyncio
async def test_run_snapshot_starts_with_all_stages_visible(tmp_path: Path) -> None:
    record = RunRecord.create("run-1", tmp_path)
    snapshot = record.snapshot()

    assert snapshot.status == RunStatus.RUNNING
    assert [stage.stage_id for stage in snapshot.stages] == list(StageId)
    assert snapshot.stages[0].status == StageStatus.RUNNING
    assert all(stage.status == StageStatus.PENDING for stage in snapshot.stages[1:])


@pytest.mark.asyncio
async def test_event_stream_emits_new_snapshot_after_stage_transition(tmp_path: Path) -> None:
    record = RunRecord.create("run-1", tmp_path)
    stream = record.events()
    initial = await anext(stream)

    await record.complete_stage(StageId.WORKLOAD, result_completeness="complete")
    await record.start_stage(StageId.OPERATOR_IR)
    updated = await asyncio.wait_for(anext(stream), timeout=1)

    assert initial.version == 0
    assert updated.version > initial.version
    assert updated.stages[0].status == StageStatus.COMPLETED
    assert updated.stages[1].status == StageStatus.RUNNING


@pytest.mark.asyncio
async def test_failed_stage_skips_remaining_stages(tmp_path: Path) -> None:
    record = RunRecord.create("run-1", tmp_path)

    await record.fail_stage(StageId.WORKLOAD, "invalid model")
    snapshot = record.snapshot()

    assert snapshot.status == RunStatus.FAILED
    assert snapshot.stages[0].status == StageStatus.FAILED
    assert all(stage.status == StageStatus.SKIPPED for stage in snapshot.stages[1:])
    persisted = json.loads((tmp_path / "run_state.json").read_text())
    assert persisted["status"] == "failed"
    assert persisted["stages"][1]["status"] == "skipped"
