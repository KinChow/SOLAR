"""Run and stage state machine used by HTTP and SSE interfaces."""

import asyncio
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import AsyncIterator, Dict, List, Optional

from pydantic import BaseModel, Field


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class StageStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class StageId(str, Enum):
    WORKLOAD = "stage1"
    OPERATOR_IR = "stage2"
    COST = "stage3"
    PERFORMANCE = "stage4"


_STAGE_LABELS = {
    StageId.WORKLOAD: "Workload Graph",
    StageId.OPERATOR_IR: "Hybrid Operator IR",
    StageId.COST: "Cost IR",
    StageId.PERFORMANCE: "Performance",
}


class StageState(BaseModel):
    stage_id: StageId
    label: str
    status: StageStatus = StageStatus.PENDING
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    duration_ms: Optional[float] = None
    result_completeness: Optional[str] = None
    error: Optional[str] = None


class RunSnapshot(BaseModel):
    run_id: str
    status: RunStatus
    version: int
    stages: List[StageState]
    final_result: Optional[Dict[str, object]] = None
    error: Optional[str] = None


class RunRecord:
    def __init__(self, run_id: str, run_dir: Path) -> None:
        self.run_id = run_id
        self.run_dir = Path(run_dir)
        self.status = RunStatus.RUNNING
        self.version = 0
        self.stages = [StageState(stage_id=stage, label=_STAGE_LABELS[stage]) for stage in StageId]
        self.stages[0].status = StageStatus.RUNNING
        self.stages[0].started_at = _now()
        self.final_result: Optional[Dict[str, object]] = None
        self.error: Optional[str] = None
        self._condition = asyncio.Condition()
        self._persist()

    @classmethod
    def create(cls, run_id: str, run_dir: Path) -> "RunRecord":
        return cls(run_id, run_dir)

    def snapshot(self) -> RunSnapshot:
        return RunSnapshot(
            run_id=self.run_id,
            status=self.status,
            version=self.version,
            stages=[stage.model_copy(deep=True) for stage in self.stages],
            final_result=dict(self.final_result) if self.final_result is not None else None,
            error=self.error,
        )

    def stage(self, stage_id: StageId) -> StageState:
        return next(stage for stage in self.stages if stage.stage_id == stage_id)

    async def _changed(self) -> None:
        async with self._condition:
            self.version += 1
            self._persist()
            self._condition.notify_all()

    def _persist(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "run_state.json").write_text(
            self.snapshot().model_dump_json(indent=2)
        )

    async def start_stage(self, stage_id: StageId) -> None:
        stage = self.stage(stage_id)
        if stage.status != StageStatus.PENDING:
            raise ValueError(f"stage '{stage_id.value}' is not pending")
        stage.status = StageStatus.RUNNING
        stage.started_at = _now()
        await self._changed()

    async def complete_stage(
        self,
        stage_id: StageId,
        *,
        result_completeness: str,
        duration_ms: Optional[float] = None,
    ) -> None:
        stage = self.stage(stage_id)
        if stage.status != StageStatus.RUNNING:
            raise ValueError(f"stage '{stage_id.value}' is not running")
        stage.status = StageStatus.COMPLETED
        stage.finished_at = _now()
        stage.duration_ms = duration_ms
        stage.result_completeness = result_completeness
        await self._changed()

    async def fail_stage(self, stage_id: StageId, error: str) -> None:
        stage = self.stage(stage_id)
        stage.status = StageStatus.FAILED
        stage.finished_at = _now()
        stage.error = error
        for later_stage in self.stages[self.stages.index(stage) + 1 :]:
            later_stage.status = StageStatus.SKIPPED
        self.status = RunStatus.FAILED
        self.error = error
        await self._changed()

    async def complete_run(self, final_result: Dict[str, object]) -> None:
        if any(stage.status != StageStatus.COMPLETED for stage in self.stages):
            raise ValueError("all stages must be completed before completing a run")
        self.status = RunStatus.COMPLETED
        self.final_result = dict(final_result)
        await self._changed()

    async def events(self) -> AsyncIterator[RunSnapshot]:
        last_version = -1
        while True:
            snapshot = self.snapshot()
            if snapshot.version != last_version:
                last_version = snapshot.version
                yield snapshot
                if snapshot.status in {RunStatus.COMPLETED, RunStatus.FAILED}:
                    return
            async with self._condition:
                await self._condition.wait_for(lambda: self.version != last_version)
