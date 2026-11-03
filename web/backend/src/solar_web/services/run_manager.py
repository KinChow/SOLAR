"""Create, retain and query single-machine demo runs."""

import asyncio
import json
from pathlib import Path
from typing import Dict, List
from uuid import uuid4

import yaml

from solar_web.api.schemas import RunCreateRequest
from solar_web.domain.models import RunRecord, RunSnapshot, StageId, StageStatus
from solar_web.services.artifact_reader import ArtifactReader
from solar_web.services.input_builder import ModelFileBuilder
from solar_web.workers.pipeline_worker import PipelineWorker


class UnknownHardwareError(ValueError):
    pass


class RunNotFoundError(KeyError):
    pass


class StageResultUnavailableError(RuntimeError):
    pass


class RunBusyError(RuntimeError):
    pass


class RunManager:
    def __init__(
        self,
        runtime_root: Path,
        solar_root: Path,
        *,
        worker: PipelineWorker | None = None,
    ) -> None:
        self.runtime_root = Path(runtime_root)
        self.solar_root = Path(solar_root)
        self.worker = worker or PipelineWorker()
        self._records: Dict[str, RunRecord] = {}
        self._tasks: Dict[str, asyncio.Task[None]] = {}

    def list_hardware(self) -> List[Dict[str, object]]:
        results = []
        for path in sorted((self.solar_root / "configs" / "arch").glob("*.yaml")):
            with path.open() as stream:
                config = yaml.safe_load(stream) or {}
            hardware_id = path.stem
            results.append(
                {
                    "id": hardware_id,
                    "name": str(config.get("name") or hardware_id),
                    "validated": hardware_id == "H100_PCIe",
                }
            )
        return results

    async def create_run(self, request: RunCreateRequest) -> RunSnapshot:
        hardware_ids = {item["id"] for item in self.list_hardware()}
        if request.hardware not in hardware_ids:
            raise UnknownHardwareError(
                f"unknown hardware configuration: {request.hardware}"
            )
        if any(not task.done() for task in self._tasks.values()):
            raise RunBusyError("another analysis run is currently active")

        run_id = uuid4().hex
        run_dir = self.runtime_root / run_id
        model_file = ModelFileBuilder().build(request, run_dir / "source" / "model.py")
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "request.json").write_text(
            json.dumps(request.model_dump(mode="json"), indent=2, sort_keys=True)
        )
        record = RunRecord.create(run_id, run_dir)
        self._records[run_id] = record
        self._tasks[run_id] = asyncio.create_task(
            self.worker.run(record, request, model_file)
        )
        return record.snapshot()

    def get_record(self, run_id: str) -> RunRecord:
        try:
            return self._records[run_id]
        except KeyError as error:
            raise RunNotFoundError(run_id) from error

    def get_snapshot(self, run_id: str) -> RunSnapshot:
        return self.get_record(run_id).snapshot()

    def get_stage_result(self, run_id: str, stage_id: StageId) -> Dict[str, object]:
        record = self.get_record(run_id)
        if record.stage(stage_id).status != StageStatus.COMPLETED:
            raise StageResultUnavailableError(
                f"stage '{stage_id.value}' has not completed"
            )
        return ArtifactReader(record.run_dir).read_stage(stage_id)
